# MSRF-NAFNet GUI - Star Removal

GUI desktop per rimuovere stelle dalle immagini astronomiche.

## 🚀 Installazione

```bash
pip install -r requirements.txt
```

## 📖 Utilizzo

```bash
python gui_inference.py
```

### Steps:

1. **Seleziona Model Type**: `msrf_nafnet_s`, `msrf_nafnet_m` o `msrf_nafnet_l`
2. **Carica Checkpoint**: Seleziona il file `.pth` dal training
3. **Click "Load Model"**: Carica il modello (ottimizzato per Apple Silicon MPS)
4. **Seleziona Input Image**: Immagine con stelle da rimuovere
5. **Seleziona Output Path**: Dove salvare il risultato
6. **Imposta parametri**:
   - **Tile Size**: 512 (default) - dimensione tile per processing
   - **Overlap**: 100 (default) - overlap per blending seamless
7. **Click "Remove Stars"**: Processa l'immagine!

## ⚙️ Parametri

- **Tile Size**: 512px funziona bene per la maggior parte dei casi
- **Overlap**: 100px garantisce transizioni invisibili tra tile
  - Più alto = blending più smooth ma processing più lento
  - Minimo consigliato: 64px

## 🎯 Features

- ✅ Ottimizzato per **Apple Silicon (MPS)**
- ✅ **Tiling intelligente** con smooth blending
- ✅ **Cosine tapering** agli edges per transizioni invisibili
- ✅ Supporta immagini di qualsiasi dimensione
- ✅ Progress bar per feedback visuale
- ✅ Auto-detect del miglior device (MPS/CUDA/CPU)

## 📝 Note

- Il blending con overlap 100px garantisce che le giunzioni tra tile siano completamente invisibili
- Per immagini molto grandi, aumenta tile_size se hai RAM sufficiente
- Il modello usa EMA weights se disponibili nel checkpoint per qualità superiore
