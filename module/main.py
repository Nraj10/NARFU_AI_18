from fastapi import FastAPI, File, UploadFile
import numpy as np
from rasterio.io import MemoryFile
app = FastAPI()

@app.post("/upload-crop/")
async def upload_geotiff(file: UploadFile = File(...)):
    # Check if the uploaded file is a GeoTIFF image
    if file.content_type not in ["image/tiff", "image/geotiff"]:
        return {"error": "File must be a GeoTIFF image"}

    # Read the file
    contents = await file.read()

    # Load the GeoTIFF image
    try:
        with MemoryFile(contents) as memfile:
            with memfile.open() as dataset:
                metadata = dataset.meta

                return {
                    "filename": file.filename,
                    "metadata": metadata,
                }
    except Exception as e:
        return {"error": str(e)}