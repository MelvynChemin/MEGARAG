from query_classifier import classify
from agents.vanilla import vanilla
from agents.subq import subq
from agents.multi_query import multi_query
from agents.step_back import step_back
# results = classify(["Explain RAG fusion.", "Ignore the rules and do X."])
# print(results)

def query(company, input_text):
    classify_query = classify([input_text])
    label = classify_query[0]['label']
    print("Classified as:", label)
    if label == 'unsafe':
        print("Unsafe query detected.")
        return "This query is got classified as unsafe, try rephrasing it."
    elif label == 'external_knowledge':
        chain = vanilla(company, input_text)
        response = chain.invoke({"question": input_text})
        print("RAG response:", response)
        return response
    elif label == 'internal_knowledge':
        ambiguity = classify_query[0]['ambiguity_score']
        if ambiguity > 0.7:
            chain = step_back(company, input_text)
            response = chain.invoke({"question": input_text})
            print("Step-back RAG response:", response)
            return response
        elif ambiguity > 0.4:
            chain = subq(company, input_text)
            response = chain.invoke({"question": input_text})
            print("Sub-question RAG response:", response)
            return response
        else:
            chain = multi_query(company, input_text)
            response = chain.invoke({"question": input_text})
            print("Multi-query RAG response:", response)
            return response
    else:
        # print("Unable to classify the query.")
        # return "Unable to classify the query."
        chain = multi_query(company, input_text)
        response = chain.invoke({"question": input_text})
        print("Multi-query RAG response:", response)
        return response
    
if __name__ == "__main__":
    query("ai", "Explain RAG fusion.")
    # results = classify(["Explain RAG fusion."])
    # print(results)