# FYSearch
<img width="2560" height="1086" alt="image" src="https://github.com/user-attachments/assets/c6f451fa-5c90-42f9-87b0-a05c4df9ab46" />
<img width="2560" height="1086" alt="image" src="https://github.com/user-attachments/assets/32f4756b-ccb7-4e14-ba29-c1f1da64825f" />
<img width="2560" height="1086" alt="image" src="https://github.com/user-attachments/assets/63271bea-72b3-437b-82b4-6c8167db57da" />

[Report.pdf](https://github.com/rachit9876/fysearch/blob/main/Documentation/report.pdf) 

> **A fully offline, CPU-only multimodal forensic search engine.**
> Point it at a folder of images, PDFs, or documents — then search by text *or* by uploading an image. Everything stays on your machine. No cloud, no GPU, no internet required.

---

## What Can It Do?

| Search Type | How It Works | Example |
|-------------|-------------|---------|
| **Text → Text** | Type a query, find documents with matching *meaning* (not just keywords). | Search `"invoice fraud"` → finds documents about billing scams, even if they don't contain those exact words |
| **Text → Image** | Type a description, find *visually matching* images. | Search `"sunset over mountains"` → finds landscape photos |
| **Image → Image** | Upload an image, find *visually similar* images. | Upload a photo → find duplicates or near-matches |
| **PDF Search** | Scanned or digital PDFs are auto-extracted (OCR + text). | Search inside scanned government documents |

### How It Works (Under the Hood)

```
Your Files → Ingest → Extract Text (OCR/PDF) → Generate Embeddings → Build Vector Index
                                                                              ↓
                                              Search Query → Embed Query → Find Nearest Vectors → Results
```

- **Embeddings** = turning text/images into numerical vectors that capture *meaning*
- **Vector search** = finding the closest vectors (most similar content) using FAISS or brute-force
- **OCR** = extracting text from images/scanned PDFs using Tesseract

---

## Quick Start (5 Minutes)

### Step 1: Install System Dependencies

You need **Tesseract OCR** and **Poppler** (for PDF rendering) installed on your system.

<details>
<summary><strong>🐧 Linux / WSL (Ubuntu/Debian)</strong></summary>

```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

</details>

<details>
<summary><strong>🪟 Windows (Native)</strong></summary>

**Using Chocolatey (recommended):**
```powershell
choco install tesseract poppler
```

**Manual:**
- Tesseract: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Poppler: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)

> ⚠️ Add both to your system PATH after manual installation.

</details>

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
brew install tesseract poppler
```

</details>

---

### Step 2: Clone & Set Up Python Environment

```bash
git clone https://github.com/rachit9876/finalYear.git
cd finalYear
```

<details>
<summary><strong>🐧 Linux / WSL / 🍎 macOS</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

<details>
<summary><strong>🪟 Windows (PowerShell)</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If you get an execution policy error: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

</details>

---

### Step 3: Install Python Dependencies

```bash
pip install -e ".[web,embeddings,faiss,ocr,pdf_images]"
```

> ⏱️ This downloads the CLIP model (~350 MB) on first run. After that, everything is offline.

---

### Step 4: Initialize & Launch

```bash
fysearch init
fysearch web
```

Open **http://127.0.0.1:5000** in your browser. That's it! 🎉

---

## Using the Web UI

### 1. Set Your Dataset Folder

In the sidebar, paste the **full path** to a folder containing your files:

| Platform | Example Path |
|----------|-------------|
| Linux / WSL | `/mnt/c/Users/YourName/Pictures/test` |
| Windows | `C:\Users\YourName\Pictures\test` |
| macOS | `/Users/YourName/Pictures/test` |

> ✅ **WSL users**: You can paste either Windows paths (`C:\Users\...`) or WSL paths (`/mnt/c/Users/...`) — both are auto-converted.

Check **"Run full pipeline"** and click **Apply**. This will:
1. **Ingest** — Copy files to the local store
2. **Extract** — Run OCR on images, extract text from PDFs
3. **Build Indexes** — Create searchable vector indexes

> ⏱️ **Timing**: ~1-2 sec/file for ingestion, ~4 sec/image for embedding, ~2 sec/document for text embedding.

### 2. Search

| Tab | What to Do |
|-----|-----------|
| **Text Search** | Type any query → click Search. Uses `Auto (Smart)` mode by default — searches both text and images. |
| **Image Search** | Drag & drop an image, paste from clipboard, or click to browse → click Search. |

### 3. Advanced Options

- **Search Mode**: `Auto` (both), `Text → Text`, or `Text → Image`
- **Results**: Number of results to return (1–50)

### 4. Build Indexes Separately

If you add new files later, you can rebuild indexes from the sidebar without re-ingesting:
- **Build Image Index** — Re-embed all images
- **Build Text Index** — Re-embed all extracted text

---

## CLI Reference

All features are also available from the command line:

```bash
# Initialize project
fysearch init

# Show/edit config
fysearch config
fysearch config --text-model "sentence-transformers/clip-ViT-B-32"
fysearch config --ocr-languages "eng+hin"

# Ingest files
fysearch ingest /path/to/your/files

# Extract text (OCR + PDF parsing)
fysearch extract

# Build search indexes
fysearch build-index --modality text
fysearch build-index --modality image

# Search
fysearch search-text "your query" --top-k 10
fysearch search-text "your query" --modality image    # text → image search
fysearch search-image /path/to/query.jpg --top-k 5    # image → image search

# Launch web UI
fysearch web                          # http://127.0.0.1:5000
fysearch web --port 8000              # custom port
fysearch web --host 127.0.0.1         # localhost only
```

---

## Configuration

Settings are stored in `fysearch.config.json` at the project root:

| Key | Default | Description |
|-----|---------|-------------|
| `text_model` | `sentence-transformers/clip-ViT-B-32` | Model for text embeddings |
| `image_model` | `sentence-transformers/clip-ViT-B-32` | Model for image embeddings |
| `dataset_path` | `""` | Path to your dataset folder |
| `ocr_languages` | `eng` | Tesseract languages (e.g., `eng`, `hin`, `eng+hin`) |
| `embedding_dim` | `512` | Vector dimension (must match model output) |
| `max_workers` | `0` | Parallel workers (`0` = auto-detect from CPU cores) |

---

## For WSL Users (Windows Subsystem for Linux)

WSL is the **recommended** way to run FYSearch on Windows. Here's the full setup:

```bash
# 1. Install system deps inside WSL
sudo apt update && sudo apt install tesseract-ocr poppler-utils python3-venv

# 2. Navigate to the project (your Windows files are under /mnt/c/)
cd /mnt/c/Users/YourName/Documents/GitHub/finalYear

# 3. Create venv & install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[web,embeddings,faiss,ocr,pdf_images]"

# 4. Initialize and run
fysearch init
fysearch web
```

Then open **http://127.0.0.1:5000** in your **Windows** browser (Edge, Chrome, etc.).

> 💡 **Tip**: The web server binds to `0.0.0.0` by default, so it's automatically accessible from your Windows browser.

> 💡 **Tip**: You can paste Windows-style paths (`C:\Users\...`) in the web UI — they're auto-converted to WSL format.

---

## Project Structure

```
finalYear/
├── src/fysearch/          # Source code
│   ├── cli.py             # Command-line interface (Typer)
│   ├── config.py          # Configuration management
│   ├── db.py              # SQLite database layer
│   ├── embeddings.py      # CLIP text/image embedders
│   ├── extract.py         # OCR + PDF text extraction
│   ├── ingest.py          # File ingestion pipeline
│   ├── paths.py           # Path resolution + WSL support
│   ├── vector_index.py    # FAISS / brute-force vector search
│   ├── webapp.py          # Flask web application
│   └── templates/
│       └── index.html     # Web UI template
├── data/                  # Runtime data (auto-created)
│   ├── input/             # Original input files
│   ├── store/             # Content-addressed file store
│   ├── db/                # SQLite database
│   └── index/             # Vector indexes (.npz)
├── fysearch.config.json   # Configuration file
└── pyproject.toml         # Python package definition
```

---

## Supported File Types

| Type | Extensions | Processing |
|------|-----------|------------|
| **Images** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff` | OCR (optional) + image embedding |
| **PDFs** | `.pdf` | Text extraction + per-page image rendering + OCR for scanned pages |
| **Text** | `.txt` | Direct text reading |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `tesseract not found` | Install Tesseract and ensure it's in your PATH |
| `poppler not found` | Install Poppler and ensure it's in your PATH |
| `No module named 'sentence_transformers'` | Run `pip install -e ".[embeddings]"` |
| `No module named 'flask'` | Run `pip install -e ".[web]"` |
| Virtual env won't activate (Windows) | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Port already in use | Use `fysearch web --port 8080` |
| WSL: folder not found | Use `/mnt/c/Users/...` format for Windows paths |
| Slow first run | Normal — the CLIP model (~350 MB) downloads on first use |
| Out of memory | Reduce image count or close other apps (needs ~4-8 GB RAM for embedding) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | [CLIP ViT-B/32](https://huggingface.co/sentence-transformers/clip-ViT-B-32) via sentence-transformers |
| Vector Search | [FAISS](https://github.com/facebookresearch/faiss) (CPU) with brute-force fallback |
| OCR | [Tesseract](https://github.com/tesseract-ocr/tesseract) via pytesseract |
| PDF Rendering | [Poppler](https://poppler.freedesktop.org/) via pdf2image |
| Web UI | [Flask](https://flask.palletsprojects.com/) |
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) |
| Database | SQLite (WAL mode for concurrency) |

---

## License

MIT — see [LICENSE](LICENSE).
