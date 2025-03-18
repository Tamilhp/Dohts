from dohts.ingest_data import IngestData
import typer

app = typer.Typer()

@app.command()
def run_pipeline(knowledge_dir: str = typer.Argument(None), embedding_model: str = typer.Argument(None),
                 vector_db: str = typer.Argument(None)) -> None:
    """
        This is the pipeline that loads all the data from various sources, chuncks them
        and loads them into the vector database
    """
    pipeline = IngestData(knowledge_dir, embedding_model, vector_db)
    pipeline()


def main():
    app()

# if __name__ == "__main__":
#     app()