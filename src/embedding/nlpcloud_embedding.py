import nlpcloud

class NLPCloudEmbedding:

  def __init__(self, nlpcloud_api_key):
    self.client = nlpcloud.Client("paraphrase-multilingual-mpnet-base-v2", nlpcloud_api_key)

  # Get the list of embeddings for all messages in a channel
  def embed_channel_messages(self, channel_messages):
    msg_list = channel_messages.astype(str).tolist()

    try:
      embeddings = self.client.embeddings(msg_list) # Returns json object {embeddings: [..] }
    except:
      return "Max no. of requests per minute reached. Try again after a minute."

    return embeddings['embeddings']

  # Get the corresponding embedding for the user's query
  def embed_query(self, query):
    try:
      embedding = self.client.embeddings([query]) # Returns json object {embeddings: [..] }
    except:
      return "Max no. of requests per minute reached. Try again after a minute."

    return embedding['embeddings']