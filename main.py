import base64
import io
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from fastapi import FastAPI, Form, Header, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.processor import Processor
from src.qdrant_client import Client
from src.utils import embed_data, search

# Create the FastAPI instance
app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("imagefolder", data_dir="dataset", split="train")

# Create processor
processor = Processor(device)

# Create client and collection 
client = Client(
    "Advertisement_dataset"
)  # If you want to use localhost set cloud to False

client.create_collection()


@app.on_event("startup")
async def startup_event():
    """
    Embeds the dataset with a specified processor and client
    on the startup of the app.
    """
    embed_data(dataset, processor, client)


def launch_search(
    search_query: str, processor: Processor, client: Client, dataset: List[str]
) -> Tuple:
    """
    Performs a search query using a specified search query.

    Args:
        - search_query (str): The search query.
        - processor (Processor): The processor instance.
        - client (Client): The client instance.
        - dataset (List[str]): The dataset list.

    Returns:
        - Tuple: A tuple containing lists of corresping images and their similarity scores.
    """

    images, scores = search(search_query, processor, client, dataset)

    return images, scores


@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    """
    Handles the GET request for the index page.

    Args:
        - request (Request): The request object.

    Returns:
        - TemplateResponse: HTML response.
    """
    context = {"request": request}
    return templates.TemplateResponse("index.html", context)


@app.get("/EDA", response_class=HTMLResponse)
async def EDA(request: Request):
    """
    Handles the GET request for the EDA page.

    Args:
        - request (Request): The request object.

    Returns:
        - TemplateResponse: HTML response.
    """
    context = {"request": request}
    return templates.TemplateResponse("EDA.html", context)


@app.post("/index", response_class=HTMLResponse)
async def index_post(
    request: Request,
    hx_request: Optional[str] = Header(None),
    search_query: Optional[str] = Form(...),
):
    """
    Handles the POST request for the index page.
    performs the search, and returns the corresponding images
    from the dataset according to its ID with its corresponding score.

    Args:
        - request (Request): The request object.
        - hx_request (Optional[str]): Optional header.
        - search_query (Optional[str]): Optional form field.

    Returns:
        - TemplateResponse: HTML response.
    """

    print(search_query)
    images, scores = launch_search(search_query, processor, client, dataset)
    img = images[0]
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=img.format)
    img_bytes = img_bytes.getvalue()

    img_base64 = base64.b64encode(img_bytes).decode()
    image_score = scores[0]

    context = {
        "request": request,
        "image_base64": img_base64,
        "image_score": image_score,
    }

    if hx_request:
        return templates.TemplateResponse("images.html", context)
    return templates.TemplateResponse("index.html", context)
