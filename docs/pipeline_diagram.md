# P&ID Symbol Detection Pipeline

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        IMG["P&ID Image\n(JPG/PNG/TIFF)"]
    end

    subgraph STAGE1["Stage 1: Symbol Detection (YOLO)"]
        S1_TRAIN["Train class-agnostic YOLO\non annotated patches"]
        S1_INF["SAHI sliced inference\n(1024x1024 patches, overlap)"]
        S1_OUT["Symbol bounding boxes\n(YOLO .txt format)"]
        S1_TRAIN --> S1_INF --> S1_OUT
    end

    subgraph STAGE2["Stage 2: Symbol Classification (Few-Shot)"]
        S2_CROP["Crop detected symbols"]
        S2_EMBED["Triplet network embedding\n(support set per class)"]
        S2_MATCH["Nearest-neighbor classification\n(euclidean distance)"]
        S2_OUT["Classified symbols\n+ confidence"]
        S2_CROP --> S2_EMBED --> S2_MATCH --> S2_OUT
    end

    subgraph STAGE3["Stage 3: Pipe Detection (Classical CV)"]
        direction TB

        subgraph PREPROCESS["Preprocessing"]
            MSER["MSER text detection\n(character regions → clusters)"]
            TMASK["Text mask\n(binary)"]
            SMASK["Symbol mask\nfrom Stage 1 boxes"]
            BIN["Binarize image\n(threshold → dark features)"]
            EXCLUDE["Remove text + symbols\nfrom binary mask"]
            MORPH["Morphological line extraction\n(H/V kernels)"]
            MSER --> TMASK --> EXCLUDE
            SMASK --> EXCLUDE
            BIN --> EXCLUDE --> MORPH
        end

        subgraph LINES["Line Detection"]
            HOUGH["Hough Transform\n(probabilistic)"]
            ORTHO["Filter near-orthogonal\n(H/V only)"]
            SNAP["Snap to exact H/V"]
            MERGE1["Merge collinear segments\n(union-find clustering)"]
            BRIDGE["Bridge symbol gaps\n(connect across masked symbols)"]
            MERGE2["Re-merge after bridging"]
            HOUGH --> ORTHO --> SNAP --> MERGE1 --> BRIDGE --> MERGE2
        end

        subgraph GRAPH["Graph Construction"]
            JUNC["Cluster endpoints → junctions\n(KDTree)"]
            CLASSIFY["Classify junctions\n(T / L / cross / endpoint)"]
            SIMPLIFY["Simplify: remove\npass-through junctions"]
            JUNC --> CLASSIFY --> SIMPLIFY
        end

        subgraph LABEL["Label-First Filtering"]
            OCR["pytesseract OCR\n(read all text)"]
            REGEX["Filter diameter labels\n(@3\", O75, D110, 8\")"]
            MATCH["Match labels → nearest pipe\n(perpendicular distance)"]
            FILTER["Keep only components\nwith diameter labels"]
            OCR --> REGEX --> MATCH --> FILTER
        end

        MORPH --> HOUGH
        MERGE2 --> JUNC
        SIMPLIFY --> OCR
    end

    subgraph OUTPUT["Output"]
        VIS["Visualizations\n(pipe mask, lines, graph)"]
        JSON["pipe_graph.json\n(junctions + pipes + labels)"]
    end

    IMG --> STAGE1
    IMG --> STAGE3
    S1_OUT --> STAGE2
    S1_OUT --> SMASK
    FILTER --> VIS
    FILTER --> JSON

    style STAGE1 fill:#1a3a5c,stroke:#4a9eff,color:#fff
    style STAGE2 fill:#1a3a5c,stroke:#4a9eff,color:#fff
    style STAGE3 fill:#1a3a5c,stroke:#4a9eff,color:#fff
    style PREPROCESS fill:#2d1a3a,stroke:#9b59b6,color:#fff
    style LINES fill:#1a3a2d,stroke:#27ae60,color:#fff
    style GRAPH fill:#3a2d1a,stroke:#e67e22,color:#fff
    style LABEL fill:#3a1a1a,stroke:#e74c3c,color:#fff
    style INPUT fill:#333,stroke:#888,color:#fff
    style OUTPUT fill:#333,stroke:#888,color:#fff
```
