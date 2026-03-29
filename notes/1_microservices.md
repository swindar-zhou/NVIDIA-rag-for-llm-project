## This course is provided by NVIDIA Deep Learning Institute

This section covers setup requirements, workflow, and things to pay attention to. To complete all units, you can access through Google Colab, but you need the DLI class environment. Turn off the class env when not running tests to avoid wasting resources.

## notebook 1: microservices

## learning objectives
1. understand the class env setup and org through container-based microservices
2. learn how to interact with microservices through Jupyter Labs endpoints

## thinking questions
1. what types of resources do you want in a cloud env, and how does it differ from local?
2. if one microservice is running on another localhost, what changes?
3. how hard is it to visit a local service like a remote host — what problems does it create?
4. which microservices suit a new instance per user, and which should run continuously?

## the big picture

```
your browser
    ↓ HTTPS
remote host (AWS)
    ├── Jupyter Labs (port 9010-9012)     ← you write code here
    ├── docker_router (port 8070)         ← internal API gateway
    ├── frontend (port 8090)              ← Gradio web interface
    ├── llm_client (port 9000)            ← LLM service
    └── nginx (reverse proxy)            ← hides backend ports
```

all components run in Docker containers on the same host, talking to each other over an internal Docker network.

## Part 1: containers

NVIDIA DLI allocates AWS resources: specialized CPU/GPU, a pre-packaged OS, and public endpoints. on top of that, a series of microservices is deployed.

**microservices** — autonomous services executing specific functionality through lightweight connection protocols. each one is containerized with Docker.

**why Docker:**
- **portability**: transfer and deploy across different environments without reconfiguring
- **isolation**: each container operates independently — no service conflicts
- **scalability**: spin up or down individual services as demand changes

**limitations to keep in mind:**
- hardware sensitivity: GPU-dependent services behave differently on different hardware
- environmental factors: network latency, storage I/O can affect performance

## Part 2: Jupyter Labs

Jupyter Labs is itself a container, defined in the `docker-compose.yml`:

```yaml
lab:
  container_name: jupyter-notebook-server
  build:
    context: ..
    dockerfile: composer/Dockerfile
  ports:
    - "9010:9010"
    - "9011:9011"
    - "9012:9012"
```

the `ports` mapping (`host:container`) exposes container ports to the outside world. so when you visit `<your-endpoint>:9010`, it routes to port 9010 inside the container.

## Part 3: container interaction

from **outside** containers (host shell):

```bash
# list all running containers
docker ps -a

# copy a file out of a container
docker cp jupyter-notebook-server:/dli/task/paint-cat.jpg .

# run a command inside a container
docker exec -it jupyter-notebook-server /bin/bash -c "ls"

# tail container logs
docker logs jupyter-notebook-server
```

from **inside** Jupyter (Python), you can't access Docker directly — but you can reach other services via HTTP through `docker_router`:

```bash
# check the router's available routes
!curl -v docker_router:8070/help
# → {"Options":"[/containers, /containers/{container_id}/logs, containers/{container_id}/restart]"}
```

```python
import requests

# list all containers
requests.get("http://docker_router:8070/containers").json()

# print only running ones
for entry in requests.get("http://docker_router:8070/containers").json():
    if entry.get("status") == 'running':
        print(entry.get("name"))
```

typical output:
```
chatbot
frontend
nginx
chatbot_rproxy
frontend_rproxy
docker_router
jupyter-notebook-server
dozzle
llm_client
assessment_1
dli-logserv
```

## Part 4: the frontend service

the `frontend` container runs a Gradio interface at port 8090. two ways to access it:

**1. direct port access**
- modify the URL to port 8090: `http://<...>.courses.nvidia.com:8090`
- can work but may hit browser/firewall restrictions

**2. reverse proxy access (preferred)**
- access through port 8091 (nginx reverse proxy)
- benefits: hides backend port, better security, enables load balancing, handles SSL

generate a clickable link from Jupyter:
```javascript
%%js
var url = 'http://'+window.location.host+':8090';
element.innerHTML = '<a style="color:green;" target="_blank" href='+url+'><h1>< Link To Gradio Frontend ></h1></a>';
```

verify the frontend is up before trying to access it:
```bash
!curl -v frontend:8090
```
