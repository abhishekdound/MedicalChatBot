from duckduckgo_search import DDGS


class Web_search:
    def search_ans(self, query):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(r["body"])
        return "\n\n".join(results)

    def web_search(self, question):
        web_context=self.search_ans(question)


        web_prompt = f"""
        You are a helpful medical assistant.
        
        Answer using the web information below.
        
        Web Context:
        {web_context}
        
        Question:
        {question}
        """
        return web_prompt
