import grpc
from concurrent import futures
import logging
import vector_service_pb2_grpc
import vector_service_pb2


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
        response = vector_service_pb2.GetSimilarityResponse()
        return response


def serve():
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
