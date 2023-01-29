import json
from enums.preprocessing_stage import PreprocessingStage
from preprocessors.image_enhancer import ImageEnhancer
from preprocessors.image_filterer import ImageFilterer
from preprocessors.image_segmentor import ImageSegmentor
import os


def get_preprocessors_to_execute_for_request(request: dict):
  processing_config = request['config']
  preprocessors_required = []

  is_preprocessing_stage_enabled = lambda \
        preprocessing_stage: processing_config.get(preprocessing_stage.name,
                                                   {}).get('enabled', False)
  # TODO: if any config parameters required for preprocessors, parse and create instances
  if is_preprocessing_stage_enabled(PreprocessingStage.enhancement):
    preprocessors_required.append(ImageEnhancer())
  if is_preprocessing_stage_enabled(PreprocessingStage.filtration):
    preprocessors_required.append(ImageFilterer())
  if is_preprocessing_stage_enabled(PreprocessingStage.segmentation):
    preprocessors_required.append(ImageSegmentor())

  return preprocessors_required


def file_key_for_stage_output(filepath: str, stage: PreprocessingStage):
  filepath_without_extension = os.path.splitext(filepath)[0]
  file_extension = os.path.splitext(filepath)[1]
  return filepath_without_extension + f"_output_{stage.name}" + file_extension


def error_formatter(error) -> dict:
  return {
    'type': error.__class__.__name__
  }


def process_request(request: dict):
  request_id = request['request-id']
  img_to_process = request['image']
  preprocessors_to_execute = get_preprocessors_to_execute_for_request(request)

  # executing preprocessors
  request['output'] = {}
  input_img_for_preprocessor = img_to_process
  for preprocessor in preprocessors_to_execute:
    preprocessing_stage = preprocessor.get_preprocessing_stage()
    request['output'][preprocessing_stage.name] = {
      "image": None,
      "error": None
    }
    output_save_at = file_key_for_stage_output(img_to_process,
                                               preprocessing_stage)
    try:
      preprocessor.process(input_img_for_preprocessor, output_save_at)
      request['output'][preprocessing_stage.name]['image'] = output_save_at
      input_img_for_preprocessor = output_save_at
    except Exception as e:
      request['output'][preprocessing_stage.name]['error'] = error_formatter(e)


def lambda_handler(event, context):
  # TODO: json schema validation to implement here before processing requests

  requests = event.get('requests', [])
  for request in requests:
    process_request(request)

  return {
    'statusCode': 200,
    'body': json.dumps(requests)
  }
