## OCR Service en Kubernetes

Esta carpeta contiene los manifiestos para llevar `ocr-service` al cluster.

Estos archivos se preparan en local, pero no están pensados para ejecutarse aquí. El flujo real es:

1. construir y publicar una imagen nueva,
2. ajustar manifiestos,
3. copiar esta carpeta al entorno que sí tenga `kubectl`,
4. aplicar desde el cluster.

## Arquitectura actual

Con los últimos cambios, el proyecto ya no es solo `ocr-api + ocr-worker`. En Kubernetes ahora quedan 5 componentes:

- `ocr-api`
  API OCR principal.

- `ocr-worker`
  Worker que consume la cola de Munis, procesa PDFs y publica artefactos.
  Este es el componente que debe ir a `nodo3` y luego usar GPU.

- `ocr-search-indexer`
  Lee artefactos textuales desde `/nfs-cache` y los indexa en OpenSearch.

- `ocr-search-api`
  API interna de búsqueda sobre OpenSearch.

- `opensearch`
  Motor de búsqueda con persistencia propia.

## Qué volúmenes necesita el proyecto

### `/data`

Se usa para:

- `tmp`, `out`, `classify`
- modelos OCR
- artefactos de trabajo del worker

En los manifiestos quedó como PVC `ocr-data`.

### `/nfs-cache`

Se usa para:

- publicar artefactos OCR desde el worker
- leer artefactos textuales desde el indexador

En los manifiestos quedó como PVC `ocr-shared-cache`.

Importante:

- `ocr-worker` sí monta `/nfs-cache`
- `ocr-search-indexer` sí monta `/nfs-cache`
- `ocr-api` no lo monta
- `ocr-search-api` no lo monta
- `opensearch` no lo usa

### almacenamiento de OpenSearch

OpenSearch no debe usar el NFS OCR. Tiene almacenamiento propio vía `volumeClaimTemplates` en el `StatefulSet`.

## Estructura

```text
ocr-service/
  base/
    00-namespace.yaml
    01-configmap-ocr.yaml
    02-configmap-search.yaml
    03-secret.example.yaml
    04-service-api.yaml
    05-deployment-api.yaml
    06-deployment-worker.yaml
    07-service-search-api.yaml
    08-deployment-search-api.yaml
    09-deployment-search-indexer.yaml
    10-service-opensearch.yaml
    11-statefulset-opensearch.yaml
    kustomization.yaml
  storage/
    00-pvc-data.yaml
    01-pvc-shared-cache.yaml
    kustomization.yaml
  overlays/
    prod/
      10-patch-api.yaml
      11-patch-worker-placement.yaml
      kustomization.yaml
```

## Qué hace cada archivo

### Base

- `00-namespace.yaml`
  Namespace `ocr-service`.

- `01-configmap-ocr.yaml`
  Variables del dominio OCR.
  Lo usan `ocr-api` y `ocr-worker`.

- `02-configmap-search.yaml`
  Variables del dominio de búsqueda.
  Lo usan `ocr-search-api` y `ocr-search-indexer`.

- `03-secret.example.yaml`
  Plantilla del Secret para Munis. No debe aplicarse sin reemplazar valores.

- `04-service-api.yaml`
  Service interno para `ocr-api`.

- `05-deployment-api.yaml`
  Deployment del API OCR.

- `06-deployment-worker.yaml`
  Deployment del worker OCR.
  Incluye:
  - `initContainer` para poblar modelos
  - `emptyDir` para `/tmp`
  - `emptyDir` para `/data/tmp`
  - montaje de `/data`
  - montaje de `/nfs-cache`

- `07-service-search-api.yaml`
  Service interno para la API de búsqueda.

- `08-deployment-search-api.yaml`
  Deployment de `ocr-search-api`.

- `09-deployment-search-indexer.yaml`
  Deployment de `ocr-search-indexer`.
  Monta `/nfs-cache` en solo lectura.

- `10-service-opensearch.yaml`
  Service interno para OpenSearch.

- `11-statefulset-opensearch.yaml`
  StatefulSet de OpenSearch con persistencia propia.

### Storage

- `00-pvc-data.yaml`
  PVC `ocr-data` para `/data`.

- `01-pvc-shared-cache.yaml`
  PVC `ocr-shared-cache` para `/nfs-cache`.

### Overlay productivo

- `10-patch-api.yaml`
  Ajustes del API OCR en producción, por ahora `PUBLIC_BASE_URL`.

- `11-patch-worker-placement.yaml`
  Fuerza el worker hacia `nodo3`:
  - `nodeSelector`
  - `toleration`
  - variables NVIDIA

- `kustomization.yaml`
  Une base + storage y fija el tag de imagen.

## Mapeo de cambios desde docker-compose

Los cambios recientes de `docker-compose.yml` se tradujeron así:

- `opensearch` de compose
  pasa a `StatefulSet` + `Service`

- `ocr-search-api`
  pasa a `Deployment` + `Service`

- `ocr-search-indexer`
  pasa a `Deployment`

- `tmpfs` del worker en compose
  pasa a `emptyDir.medium: Memory` para `/tmp` y `/data/tmp`

- `opensearch_data` de compose
  pasa a almacenamiento persistente del `StatefulSet`

- `ocr_nfs_cache`
  sigue existiendo como PVC compartido para worker e indexador

## Orden lógico de aplicación

El orden conceptual queda así:

1. namespace
2. configmap OCR
3. configmap Search
4. secret
5. pvc `ocr-data`
6. pvc `ocr-shared-cache`
7. services internos
8. deployments OCR
9. deployment del indexador
10. deployment de search-api
11. statefulset de opensearch

Si usas `kustomize`, no necesitas aplicar uno por uno:

```bash
kubectl apply -k ocr-service/overlays/prod
```

Si quieres aplicar manualmente desde el cluster:

```bash
kubectl apply -f ocr-service/base/00-namespace.yaml
kubectl apply -f ocr-service/base/01-configmap-ocr.yaml
kubectl apply -f ocr-service/base/02-configmap-search.yaml
kubectl apply -f ocr-service/base/03-secret.yaml
kubectl apply -f ocr-service/storage/00-pvc-data.yaml
kubectl apply -f ocr-service/storage/01-pvc-shared-cache.yaml
kubectl apply -f ocr-service/base/04-service-api.yaml
kubectl apply -f ocr-service/base/05-deployment-api.yaml
kubectl apply -f ocr-service/base/06-deployment-worker.yaml
kubectl apply -f ocr-service/base/07-service-search-api.yaml
kubectl apply -f ocr-service/base/08-deployment-search-api.yaml
kubectl apply -f ocr-service/base/09-deployment-search-indexer.yaml
kubectl apply -f ocr-service/base/10-service-opensearch.yaml
kubectl apply -f ocr-service/base/11-statefulset-opensearch.yaml
```

## Parámetros que debes revisar antes de aplicar

- `ocr-service/base/03-secret.example.yaml`
  Reemplazar `MUNIS_BASE_URL` y `MUNIS_OCR_TOKEN`.

- `ocr-service/storage/00-pvc-data.yaml`
  Confirmar si `longhorn` es correcto para `/data`.

- `ocr-service/storage/01-pvc-shared-cache.yaml`
  Confirmar si `nfs-storage` es correcto para `/nfs-cache`.

- `ocr-service/base/11-statefulset-opensearch.yaml`
  Ajustar storage o recursos si OpenSearch va a tener mucha carga.

- `ocr-service/overlays/prod/kustomization.yaml`
  Actualizar el tag de imagen.

- `ocr-service/overlays/prod/10-patch-api.yaml`
  Si no vas a usar `PUBLIC_BASE_URL`, puedes quitarlo o dejarlo interno.

## Build de imagen

Como ahora `requirements.txt` sí cambió por `opensearch-py`, necesitas construir y publicar una imagen nueva:

```bash
docker build -f k8s/Dockerfile -t ryuk89/ocr-service:20260424-01 .
docker push ryuk89/ocr-service:20260424-01
```

Luego ese mismo tag debe quedar en:

- `ocr-service/overlays/prod/kustomization.yaml`

## GPU

La GPU sigue reservada para `ocr-worker`, no para `ocr-api`, ni para `ocr-search-api`, ni para `ocr-search-indexer`.

En estos manifiestos:

- el `worker` ya queda fijado a `nodo3`
- todavía no agregué `nvidia.com/gpu`

Cuando Kubernetes ya exponga formalmente la GPU, el cambio correcto es agregarlo en el overlay productivo del worker.

## Estado actual de la propuesta

Ya quedó resuelto:

- OCR principal
- worker en `nodo3`
- cache NFS compartido
- indexador de artefactos textuales
- API de búsqueda
- OpenSearch persistente
- adaptación del `tmpfs` de compose a `emptyDir`

Sigue pendiente de tu lado:

- secret real
- confirmar `storageClass`
- construir y publicar la nueva imagen
- aplicar desde el cluster
- más adelante, habilitar la reserva formal de GPU
