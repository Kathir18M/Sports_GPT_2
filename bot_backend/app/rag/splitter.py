class DocumentSplitter:

    def __init__(self, chunk_size=800, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def recursive_chunk(self, text: str):

        def _split(text: str, seps: list[str]):

            if len(text) <= self.chunk_size:
                return [text]

            if not seps:
                return [
                    text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)
                ]

            sep = seps[0]

            parts = text.split(sep)

            chunks = []

            current = ""

            for part in parts:

                candidate = current + sep + part if current else part

                if len(candidate) <= self.chunk_size:

                    current = candidate

                else:

                    if current:
                        chunks.append(current)

                    if len(part) > self.chunk_size:

                        chunks.extend(
                            _split(part, seps[1:])
                        )

                        current = ""

                    else:

                        current = part

            if current:
                chunks.append(current)

            return chunks

        raw_chunks = _split(text, self.separators)

        overlapped = []

        for i, chunk in enumerate(raw_chunks):

            if i == 0:

                overlapped.append(chunk)

            else:

                prev_tail = raw_chunks[i-1][-self.overlap:]

                overlapped.append(prev_tail + chunk)

        return overlapped

    def split_documents(self, documents):

        all_chunks = []

        for document in documents:

            chunks = self.recursive_chunk(document["text"])

            for chunk in chunks:

                all_chunks.append(
                    {
                        "source": document["source"],
                        "text": chunk
                    }
                )

        return all_chunks