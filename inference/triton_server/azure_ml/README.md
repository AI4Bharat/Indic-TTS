# Serving using Azure Machine Learning

## Pre-requisites

```
cd inference/triton_server
```

### Setting AML environment

Set the environment for AML:
```
export RESOURCE_GROUP=Dhruva-prod
export WORKSPACE_NAME=dhruva--central-india
export DOCKER_REGISTRY=dhruvaprod
```

Also remember to edit the `yml` files accordingly.

### Pushing the docker image to Container Registry

```
az acr login --name $DOCKER_REGISTRY
docker tag tts_triton $DOCKER_REGISTRY.azurecr.io/tts/triton-tts-coqui:latest
docker push $DOCKER_REGISTRY.azurecr.io/tts/triton-tts-coqui:latest
```

### Creating the execution environment

```
az ml environment create -f azure_ml/environment.yml -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

## Deployment

Since we have different models for different languages, to reduce the no. of deployments, we recommend that some of the models be grouped and deployed together, based on how much we can fit into the GPU RAM we're deploying on.

In our case, we group it as follows:
- [North-Indian languages](https://en.wikipedia.org/wiki/Indo-Aryan_languages)  
  - Indo-Aryan languages: `as`, `bn`, `gu`, `hi`, `mr`, `or`, `pa`, `raj`
  - Language wise folders should be placed in `inference/checkpoints/indo-aryan/checkpoints`
- [South-Indian languages](https://en.wikipedia.org/wiki/Dravidian_languages)  
  - Dravidian languages: `kn`, `ml`, `ta`, `te`
  - Language wise folders should be placed in `inference/checkpoints/dravidian/checkpoints`
- Remaining languages  
  - Miscellaneous languages: `en`, `brx`, `mni`
    - (Combination of Indian-English and [Tibeto-Burman languages](https://en.wikipedia.org/wiki/Tibeto-Burman_languages))
  - Language wise folders should be placed in `inference/checkpoints/misc/checkpoints`

In this tutorial, we show example on how to perform a deployment for North-Indian languages, the config files for which are available in the directory: `azure_ml/indo-aryan`. (For other groups, follow similarly)

### Registering the model

```
az ml model create --file azure_ml/indo-aryan/model.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
```

### Publishing the endpoint for online inference

```
az ml online-endpoint create -f azure_ml/indo-aryan/endpoint.yml -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

Now from the Azure Portal, open the Container Registry, and grant ACR_PULL permission for the above endpoint, so that it is allowed to download the docker image.

### Attaching a deployment

```
az ml online-deployment create -f azure_ml/indo-aryan/deployment.yml --all-traffic -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

### Testing if inference works

1. From Azure ML Studio, go to the "Consume" tab, and get the endpoint domain (without `https://` or trailing `/`) and an authentication key.
2. In `client.py`, enable `ENABLE_SSL = True`, and then set the `ENDPOINT_URL` variable as well as `Authorization` value inside `HTTP_HEADERS`.
3. Run `python3 client.py`
