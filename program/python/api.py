@app.post("/retrain", response_model=GenericRetrainResponse)
async def retrain(request: GenericRetrainRequest, model: Model = Depends(get_model)):
    status = model.retrain(request.texts)
    return GenericRetrainResponse(status=status)

@app.post("/generate", response_model=GenericGenerateResponse)
async def generate(request: GenericGenerateRequest, model: Model = Depends(get_model)):
    dataset = model.generate(request.num_text)
    return GenericGenerateResponse(dataset=dataset)

@app.post("/reset", response_model=GenericResetResponse)
async def reset(model: Model = Depends(get_model)):
    status = model.reset_model()
    return GenericResetResponse(status=status)

@app.post("/delete", response_model=GenericDeleteResponse)
async def reset(model: Model = Depends(get_model)):
    status = model.delete_model()
    return GenericDeleteResponse(status=status)

@app.post("/model-list", response_model=GenericModelListResponse)
async def useModel(model: Model = Depends(get_model)):
    current,models = model.get_model_list()
    return GenericModelListResponse(current=current,models=models)
    
@app.post("/use-model", response_model=GenericUseModelResponse)
async def useModel(request:GenericUseModelRequest, model: Model = Depends(get_model)):
    status = model.use_model(request.model)
    return GenericUseModelResponse(status=status)
    
