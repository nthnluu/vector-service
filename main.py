import grpc
from concurrent import futures
import logging
import vector_service_pb2_grpc
import vector_service_pb2

from datasets import load_dataset

from text_transformer.text_transformer import TextTransformer, clean_text

# Initialize text transformer
transformer = TextTransformer("justin871030/bert-base-uncased-goemotions-original-finetuned")


# VectorServiceServicer provides an implementation of the methods of the Vector service.
class VectorServiceServicer(vector_service_pb2_grpc.VectorServiceServicer):
    def GetEmbedding(self, request, context):
        """
        Retrieves the embedding for a given input.
        """
        # TODO: Not yet implemented
        response = vector_service_pb2.GetEmbeddingResponse()
        return response

    def GetSimilarity(self, request, context):
        """
        Computes and returns the similarity between two vectors.
        """
        # TODO: Not yet implemented
        transformer.compare_similarity(request.input_embedding)
        response = vector_service_pb2.GetSimilarityResponse()
        return response

    def LoadQueries(self, request, context):
        """
        Loads the queries into the text transformer.
        """
        cleaned_queries = [clean_text(query) for query in request.queries]
        transformer.load_queries(cleaned_queries)
        return vector_service_pb2.LoadQueryResponse()


def serve():
    # Start server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vector_service_pb2_grpc.add_VectorServiceServicer_to_server(
        VectorServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Listening on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
