# Contexto de decision: OCR por lotes

Fecha: 2026-02-20

## Objetivo
Aplicar OCR a un volumen grande de PDFs escaneados (aprox. 390 GB), con
prioridad en calidad y resultado profesional, aun si el procesamiento toma tiempo.

## Caracteristicas del material
- Idioma principal: espanol.
- Idiomas secundarios poco frecuentes: quechua, ingles, frances, aleman, etc.
- Documentos escaneados de calidad regular (no fotos de celular).
- La mayoria de PDFs no contiene texto.
- Tablas ocasionales.

## Decision tecnica
Se adopta OCRmyPDF + Tesseract como pipeline base (CPU).

### Por que
- Produce salida profesional con capa OCR invisible y soporte de PDF/A.
- Preserva el documento original y su estructura.
- Flujo estable, reproducible y comun en produccion para archivos oficiales.
- Mejora real de calidad posible via preprocesado (deskew, limpieza, etc.).

## GPU
No se usara GPU como base del pipeline.

### Motivo
- Motores GPU no generan PDF/A ni la capa OCR integrada de forma nativa;
  requieren reconstruir el PDF, aumentando complejidad y riesgo.
- El beneficio principal de GPU es velocidad, pero la prioridad es calidad.

## Nota operativa
Si en el futuro se necesita acelerar, se puede evaluar un flujo hibrido:
OCRmyPDF como salida oficial y un motor GPU solo para paginas dificiles o
extraccion adicional de texto.
