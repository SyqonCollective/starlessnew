# MSRF-NAFNet: Star Removal with Genuine Texture Reconstruction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green)

**Multi-Scale Receptive Field NAFNet** per la rimozione professionale delle stelle dalle immagini astronomiche, con ricostruzione di texture genuina (NO blob artifacts tipici delle U-Net).

## ✨ Caratteristiche Principali

- 🎯 **Texture Genuina**: Context aggregation e multi-scale features per riempire le stelle con vera texture dall'ambiente circostante
- ⚡ **Ottimizzato RTX 5090**: Mixed precision, torch.compile, gradient accumulation
- 🎨 **Loss Avanzate**: Combinazione di L1, Perceptual, Texture, Edge e Frequency loss
- 🔄 **Training Moderno**: EMA, warmup scheduling, gradient clipping
- 📊 **Monitoring Completo**: TensorBoard, visualizzazioni, metriche PSNR/SSIM

## 🏗️ Architettura

MSRF-NAFNet combina:
- **NAFNet-S** come base (efficiente e potente)
- **Multi-Scale Convolutions** per receptive fields diversificati
- **Channel & Spatial Attention** per preservare texture
- **Context Aggregation Modules** per raccogliere texture genuina dal contesto
- **Texture-Aware Blocks** per evitare blob artifacts

## 🚀 Quick Start

### 1. Installazione

```bash
# Clona o prepara la directory
cd /Users/michaelruggeri/Desktop/starless

# Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Prepara il Dataset

Il dataset è già organizzato:
```
starless/
├── train/
│   ├── input/    # Immagini con stelle
│   └── target/   # Immagini senza stelle (ground truth)
└── val/
    ├── input/
    └── target/
```

### 3. Training

```bash
# Training con config di default (ottimizzato RTX 5090)
python train.py --config config.yaml

# Resume da checkpoint
python train.py --config config.yaml --resume output/checkpoints/checkpoint_epoch_0100.pth
```

### 4. Inference

```bash
# Singola immagine
python inference.py \
    --model output/checkpoints/best_model.pth \
    --input test_image.png \
    --output result.png \
    --amp

# Directory intera
python inference.py \
    --model output/checkpoints/best_model.pth \
    --input test_images/ \
    --output results/ \
    --recursive \
    --amp

# Immagini molto grandi con tiling
python inference.py \
    --model output/checkpoints/best_model.pth \
    --input large_image.tif \
    --output result.tif \
    --tile-size 1024 \
    --tile-overlap 64 \
    --amp
```

## ⚙️ Configurazione

Modifica `config.yaml` per personalizzare:

### Per RTX 5090 (24GB VRAM)
```yaml
training:
  batch_size: 32  # Massimizza utilizzo GPU
  use_amp: true
  compile_model: true

data:
  patch_size: 384  # Qualità superiore
```

### Per GPU più piccole
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Simula batch_size=32
  
data:
  patch_size: 256
```

### Modelli Disponibili

- **msrf_nafnet_s**: ~2M parametri (veloce, ottimo per starting)
- **msrf_nafnet_m**: ~5M parametri (bilanciato)
- **msrf_nafnet_l**: ~12M parametri (massima qualità)

## 📊 Monitoring

Durante il training, monitora con TensorBoard:

```bash
tensorboard --logdir output/logs
```

Metriche tracciate:
- Train/Val Loss (con componenti separate)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Learning Rate
- Visualizzazioni Input/Output/Target

## 🎯 Tips per Risultati Ottimali

### 1. **Anti-Blob Strategy**
Il modello usa:
- Context aggregation per "guardare" texture circostante
- Multi-scale features per catturare pattern a diverse scale
- Texture loss (Gram matrices) per matching statistico
- Frequency loss per preservare dettagli ad alta frequenza

### 2. **Training Duration**
- Min 200-300 epochs per convergenza
- Monitor PSNR: dovrebbe raggiungere >35dB per buoni risultati
- Early stopping se validation PSNR non migliora per 50 epochs

### 3. **Data Augmentation**
Già inclusa nel dataloader:
- Flip orizzontale/verticale
- Rotazioni 90°
- Color jitter (subtle per astronomia)
- Noise realistico

### 4. **Fine-tuning**
Se hai pochi dati, parti da checkpoint pre-trained:
```bash
python train.py \
    --config config.yaml \
    --resume pretrained_model.pth
```

## 📁 Struttura del Progetto

```
starless/
├── model.py              # Architettura MSRF-NAFNet
├── dataset.py            # DataLoader ottimizzato
├── train.py              # Training loop
├── inference.py          # Script di inferenza
├── losses.py             # Loss functions avanzate
├── utils.py              # Utility functions
├── config.yaml           # Configurazione
├── requirements.txt      # Dipendenze
└── output/               # Output (creato automaticamente)
    ├── checkpoints/      # Model checkpoints
    ├── logs/             # TensorBoard logs
    └── visualizations/   # Training visualizations
```

## 🔬 Dettagli Tecnici

### Context Aggregation
Usa self-attention per raccogliere features da regioni circostanti:
```python
# Nel forward pass
x = x + self.context_agg(x)
```
Questo permette al modello di "vedere" e copiare texture genuina.

### Multi-Scale Receptive Fields
Convoluzioni parallele con kernel 1x1, 3x3, 5x5, 7x7:
```python
scales = [1, 3, 5, 7]
# Cattura pattern da fini a grossolani
```

### Mixed Precision Training
Velocità ~2x su RTX 5090:
```python
with autocast(enabled=use_amp):
    output = model(input)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```yaml
# In config.yaml
training:
  batch_size: 4  # Riduci batch size
  gradient_accumulation_steps: 8  # Compensa con accumulation
data:
  patch_size: 192  # Riduci patch size
```

### Blob Artifacts
Se vedi ancora blob:
1. Aumenta `texture_weight` in config
2. Aumenta `perceptual_weight`
3. Usa modello più grande (msrf_nafnet_m/l)
4. Training più lungo (>500 epochs)

### Training Instabile
```yaml
optimizer:
  lr: 0.0001  # Riduci learning rate

scheduler:
  warmup_steps: 2000  # Warmup più lungo
```

## 📈 Risultati Attesi

Su dataset astronomici ben preparati:
- **PSNR**: >35 dB (ottimo), >38 dB (eccellente)
- **SSIM**: >0.95
- **Qualità visiva**: Nessun blob, texture naturale, preservazione dettagli

## 🎓 Citazioni

Se usi questo codice, considera di citare:

```bibtex
@article{nafnet2022,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={ECCV},
  year={2022}
}
```

## 📝 Note

- Richiede **PyTorch 2.0+** per `torch.compile`
- Testato su **RTX 5090** (ma funziona su qualsiasi GPU moderna)
- Supporta immagini **RGB** (può essere adattato per RAW/FITS)
- **Non richiede maschere** delle stelle - lavora end-to-end

## 🤝 Support

Per problemi o domande:
1. Controlla la sezione Troubleshooting
2. Verifica i logs in `output/logs/`
3. Controlla le visualizzazioni in `output/visualizations/`

---

**Buon training! 🚀✨**

La chiave per evitare blob è il **context aggregation** + **multi-scale features** + **texture loss**. Il modello impara a "guardare" la texture circostante e copiarla in modo intelligente, invece di inventare blob generici.
