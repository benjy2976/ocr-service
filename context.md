# Contexto de decision: OCR documental con control de sellos y texto

Fecha base: 2026-02-20  
Actualizado: 2026-03-17

## Objetivo real del producto
Aplicar OCR a un volumen grande de PDFs escaneados para busqueda documental.

El entregable principal es:
- un `PDF searchable` con la imagen original intacta
- y, mas adelante, una salida textual derivada (`txt`, `md` o indice de busqueda)

La prioridad funcional no es "limpiar visualmente" el documento sino:
- extraer texto util
- evitar inventar caracteres
- reducir al minimo el texto falso generado por sellos, logos, firmas o huellas

## Criterio de calidad
Orden de gravedad de errores:
1. peor error: agregar texto falso desde sellos / logos / firmas
2. segundo error: romper una linea de texto valida por preprocesado agresivo
3. tolerable en algunos casos: perder pocos caracteres por superposicion real del sello

## Decision tecnica base
Se mantiene `OCRmyPDF + Tesseract` como pipeline oficial de salida.

La salida final sigue siendo:
- PDF searchable con imagen original intacta
- capa OCR invisible superpuesta

La version preprocesada o enmascarada es solo un artefacto de trabajo interno.

## Por que no se cambia el pipeline oficial
- OCRmyPDF produce un PDF searchable estable y profesional
- mantiene la estructura original del PDF
- evita reconstruir manualmente el PDF final
- encaja mejor con archivo institucional y preservacion

## Cambio conceptual importante
Antes:
- la idea era "quitar sellos para que OCR lea mejor"

Ahora:
- la idea es "proteger el texto valido y bloquear texto falso"

Eso implica:
- no editar visualmente el PDF final
- usar deteccion visual como apoyo
- tomar decisiones de redaccion sobre la capa OCR, no sobre la imagen final

## Estado actual del control de sellos
### Detector principal
El detector que mejor resultado esta dando es:
- `stamp_detector_v2.pt`

### Uso del detector
Se usa para:
- detectar regiones de sello / logo / firma / huella
- generar PDFs de depuracion
- decidir que partes de la capa OCR deben recortarse

### Aprendizaje clave
La heuristica basada solo en OCR ya reconocido para inferir superposicion con texto no es suficiente.
Hace falta una mejor deteccion de texto visual real.

## Estrategia actual de OCR conservador
Existe un modo `searchable_conservative` cuya intencion es:
- partir del OCR del PDF original
- detectar cajas problemáticas
- redaccionar texto OCR dentro de esas regiones
- fusionar la capa OCR filtrada sobre el PDF original

El objetivo de este modo es reducir texto espurio sin reescribir manualmente toda la capa OCR.

## Problema abierto principal
La parte mas debil del sistema actual no es la deteccion de sellos.

La parte debil es:
- detectar con precision donde hay texto real del documento
- y, por lo tanto, decidir por que lado de una caja se puede recortar agresivamente y por cual hay que ser conservador

## Decision nueva: abrir una linea paralela de deteccion de texto
Se decide crear una linea propia para detectar texto visual en pagina.

### Motivo
Hoy estamos intentando inferir el texto visual a partir del OCR.
Eso esta al reves.

El flujo correcto debe tender a ser:
1. detectar texto visual
2. detectar sellos / logos / firmas
3. cruzar ambas capas
4. decidir redaccion OCR por interseccion real

## Dataset de texto
Se reutilizan las mismas paginas base usadas para sellos, pero en una ruta separada:
- `data/out/stamp_pages` se mantiene intacto
- `data/out/text_pages` es el nuevo workspace para texto

### Estructura actual
- `images`: copia de las paginas
- `labels_auto`: cajas automáticas de texto
- `labels_reviewed`: correcciones manuales
- `previews_auto`: previews de las cajas

## Preanotacion de texto
Se implemento un flujo de preanotacion con PaddleOCR sobre `text_pages/images`.

### Decision
Se empieza con una sola clase:
- `text_block`

### Motivo
- es mas rapido de anotar
- sirve para un primer modelo
- deberia bastar para tomar decisiones de cruce sello-texto

## Limitacion operativa detectada
PaddleOCR dentro del contenedor actual esta corriendo en CPU.

Confirmado:
- `compiled_with_cuda = False`
- `device = cpu`

Por eso:
- el autolabeling de texto no usa la RTX 3060
- la mejora inmediata se hizo via multiproceso CPU y modo resumible

## Decision operativa sobre autolabeling
El script de autolabel de texto debe:
- ser resumible
- no reprocesar paginas ya hechas
- reconstruir previews faltantes sin volver a correr OCR
- permitir multiproceso en CPU

## Revision humana
Ya existen dos rutas separadas:
- revision normal de texto: `/text/review`
- revision de casos saltados: `/text/review/skipped`

### Politica de trabajo
- un usuario puede `Validar`
- puede `Saltar` casos dudosos
- los `skipped` se revisan al final
- si el usuario refresca, primero debe recuperar su item `in_process`

## Siguiente hito
Entrenar un primer detector de `text_block` sobre el dataset de texto.

No se recomienda entrenar aun si:
- no se reviso una muestra razonable de `labels_auto`
- no se confirmo que la calidad visual de las cajas sea aceptable

## Vision de arquitectura objetivo
Arquitectura deseada a mediano plazo:
1. detector de texto visual
2. detector de sellos / firmas / logos
3. logica de interseccion por lado / region
4. redaccion de capa OCR
5. fusion sobre PDF original

El PDF final no debe alterarse visualmente.
