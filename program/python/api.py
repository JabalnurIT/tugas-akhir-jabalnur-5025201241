from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .distilgpt2.model import Model, get_model

app = FastAPI()


class GenericRetrainRequest(BaseModel):
    texts: str

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "texts": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }

class GenericGenerateRequest(BaseModel):
    num_text: int

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "num_text": 5
            }
        }

class GenericResetRequest(BaseModel):
    model: str

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "model": "age_analysis",
            }
        }

class GenericUseModelRequest(BaseModel):
    model: str

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "model": "age_analysis",
            }
        }

class GenericRetrainResponse(BaseModel):
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "Success"
            }
        }

class GenericGenerateResponse(BaseModel):
    dataset: str

    class Config:
        json_schema_extra = {
            "example": {
                "dataset": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }

class GenericResetResponse(BaseModel):
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "Success"
            }
        }

class GenericDeleteResponse(BaseModel):
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "Success"
            }
        }

class GenericModelListResponse(BaseModel):
    current: str
    models: list

    class Config:
        json_schema_extra = {
            "example": {
                "current": "age_analysis",
                "model": ["age_analysis"]
            }
        }

class GenericUseModelResponse(BaseModel):
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "age_analysis"
            }
        }

        

@app.post("/retrain", response_model=GenericRetrainResponse)
async def retrain(request: GenericRetrainRequest, model: Model = Depends(get_model)):
    status = model.retrain(request.texts)
    return GenericRetrainResponse(
        status=status
    )

@app.post("/generate", response_model=GenericGenerateResponse)
async def generate(request: GenericGenerateRequest, model: Model = Depends(get_model)):
    dataset = model.generate(request.num_text)
    return GenericGenerateResponse(
        dataset=dataset
    )

@app.post("/reset", response_model=GenericResetResponse)
async def reset(model: Model = Depends(get_model)):
    status = model.reset_model()
    return GenericResetResponse(
        status=status
    )

@app.post("/delete", response_model=GenericDeleteResponse)
async def reset(model: Model = Depends(get_model)):
    status = model.delete_model()
    return GenericDeleteResponse(
        status=status
    )

@app.post("/model-list", response_model=GenericModelListResponse)
async def useModel(model: Model = Depends(get_model)):
    current,models = model.get_model_list()
    return GenericModelListResponse(
        current=current,
        models=models
    )
    
@app.post("/use-model", response_model=GenericUseModelResponse)
async def useModel(request:GenericUseModelRequest, model: Model = Depends(get_model)):
    status = model.use_model(request.model)
    return GenericUseModelResponse(
        status=status
    )
    
