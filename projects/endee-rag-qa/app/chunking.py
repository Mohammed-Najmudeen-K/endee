def chunk_text(text, size=200, overlap=40):
    words = text.split()
    step = max(size - overlap, 1)
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size])
        if piece:
            yield piece
        if i + size >= len(words):
            break