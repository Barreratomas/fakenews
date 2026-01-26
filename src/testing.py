from inference.predict import predict


data = {"type": "text", "content": "Este es un ejemplo de noticia."}
result = predict(data, method="lime")
print(result)

# url_data = {"type": "url", "content": "https://elpais.com/..." }
# result_url = predict(url_data, method="attention")
# print(result_url)
