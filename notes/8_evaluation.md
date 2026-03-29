## notebook 8: evaluating RAG systems

now that we have a working RAG system, how do we know if it's actually good? this notebook covers systematic evaluation using the **LLM-as-a-Judge** pattern.

## learning objectives
1. set up a RAG chain for evaluation (no conversation memory)
2. generate synthetic QA pairs from the document corpus
3. use LLM pairwise evaluation to score RAG output quality

## thinking questions
1. what does a "preference score" actually measure — and does it align with your real use case?
2. should you separately evaluate retrieval quality vs. generation quality?
3. how do you avoid evaluation bias when the judge LLM might favor its own output style?

## Part 1: the evaluation RAG chain

for evaluation, we strip out conversation memory (convstore) — we want to measure document retrieval + generation only, not memory effects:

```python
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

# context_getter: input string → retrieve docs → reorder → format as string
context_getter = itemgetter('input') | docstore.as_retriever() | long_reorder | docs2str
retrieval_chain = {'input': (lambda x: x)} | RunnableAssign({'context': context_getter})

# generator_chain: {'input', 'context'} → LLM answer
generator_chain = chat_prompt | llm
generator_chain = {'output': generator_chain} | RunnableLambda(output_puller)

rag_chain = retrieval_chain | generator_chain
```

`output_puller` normalizes the output to always yield strings from the `'output'` key:

```python
def output_puller(inputs):
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')
```

## Part 2: generating synthetic QA pairs

instead of hand-labeling test data, use an LLM to generate it from the documents:

```python
for i in range(num_questions):
    doc1, doc2 = random.sample(docs, 2)
    sys_msg = (
        "Use the documents provided by the user to generate an interesting question-answer pair."
        " Try to use both documents if possible, and rely more on the document bodies than the summary."
        " Use the format:\nQuestion: (good question, 1-3 sentences)\n\nAnswer: (answer derived from the documents)"
        " DO NOT SAY: 'Here is an interesting question pair'. FOLLOW FORMAT!"
    )
    qa_pair = (simple_prompt | llm).invoke({'system': sys_msg, 'input': f"Doc1: {format_chunk(doc1)}\nDoc2: {format_chunk(doc2)}"})
    synth_questions.append(qa_pair.split('\n\n')[0])   # "Question: ..."
    synth_answers.append(qa_pair.split('\n\n')[1])     # "Answer: ..."
```

**why 2 documents**: forces the LLM to generate cross-document questions — harder, more representative of real user queries on a multi-doc corpus.

## Part 3: computing RAG answers

run the same questions through the RAG chain to get its answers:

```python
rag_answers = []
for q in synth_questions:
    rag_answer = ""
    for token in rag_chain.stream(q):
        rag_answer += token
    rag_answers.append(rag_answer)
```

now we have three things for each question:
- `synth_questions[i]` — the question
- `synth_answers[i]` — the "ground truth" (LLM-generated from source docs)
- `rag_answers[i]` — the RAG system's answer

## Part 4: LLM-as-a-Judge evaluation

compare each RAG answer to the synthetic ground truth:

```python
eval_prompt = ChatPromptTemplate.from_template("""INSTRUCTION:
Evaluate the following Question-Answer pair for human preference and consistency.
Assume the first answer is a ground truth answer and has to be correct.
Assume the second answer may or may not be true.
[1] The second answer lies, does not answer the question, or is inferior to the first answer.
[2] The second answer is better than the first and does not introduce any inconsistencies.

Output Format:
[Score] Justification

{qa_trio}

EVALUATION:
""")

pref_score = []
for q, a_synth, a_rag in zip(synth_questions, synth_answers, rag_answers):
    qa_trio = f"Question: {q}\n\nAnswer 1 (Ground Truth): {a_synth}\n\nAnswer 2 (New Answer): {a_rag}"
    pref_score.append((eval_prompt | llm).invoke({'qa_trio': qa_trio}))

# aggregate: what fraction of RAG answers scored [2]?
preference_score = sum(("[2]" in score) for score in pref_score) / len(pref_score)
print(f"Preference Score: {preference_score}")
```

**interpretation**:
- `1.0` = RAG always outperforms ground truth (suspicious — might be hallucinating)
- `0.5` = RAG matches ground truth half the time (decent)
- `0.0` = RAG consistently worse than ground truth (needs improvement)

## Part 5: evaluation strategies beyond pairwise

| strategy | what it measures | when to use |
|---|---|---|
| pairwise preference | is RAG answer better than ground truth? | quick quality estimate |
| style evaluation | does output match expected tone/format? | customer-facing products |
| ground-truth accuracy | exact match on known-answer questions | factual QA tasks |
| retrieval evaluation | are the right docs being retrieved? | debugging retrieval quality |
| trajectory evaluation | multi-turn conversation quality | conversational agents |

**practical note**: always separate retrieval evaluation from generation evaluation. a RAG system can fail at retrieval (wrong docs) or generation (right docs but bad answer) — you need to know which.

## Part 6: limitations of synthetic evaluation

synthetic QA pairs (ground truth) are themselves LLM-generated, so:
- the judge LLM may prefer a similar style to the synthetic answers
- biases in the generating LLM propagate to the evaluation
- doesn't test out-of-distribution user queries

use synthetic evaluation as a **fast iteration signal**, not a final quality gate. supplement with human evaluation for production decisions.
