def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'text': chunk_text,
            'start_word': i,
            'end_word': min(i + chunk_size, len(words)),
            'word_count': len(chunk_words)
        })
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def search_in_text(text, query, context_length=100):
    if not query.strip():
        return []
    
    text_lower = text.lower()
    query_lower = query.lower()
    results = []
    
    start = 0
    while True:
        pos = text_lower.find(query_lower, start)
        if pos == -1:
            break
        
        context_start = max(0, pos - context_length)
        context_end = min(len(text), pos + len(query) + context_length)
        
        context = text[context_start:context_end]
        
        results.append({
            'position': pos,
            'context': context,
            'highlight_start': pos - context_start,
            'highlight_end': pos - context_start + len(query)
        })
        
        start = pos + 1
    
    return results
