# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import random
import shutil
import sys
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
# from scipy.misc import imread, imresize
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append('.')
import iep.utils as utils
import iep.programs
from iep.data import ClevrDataset, ClevrDataLoader
from iep.preprocess import tokenize, encode
from clevr_dataset import ClevrDataset
from torchvision import transforms
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', type=str, default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--question', default=None)
parser.add_argument('--root_dir', type=str, default='../data/CLEVR_v1.0/images')
parser.add_argument('--image_h5', type=str)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)

parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--questions_file', type=str, default=None)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_dir', default=None)

models = {
    '18k': ('program_generator_18k.pt', 'execution_engine_18k.pt'),
    '9k': ('program_generator_9k.pt', 'execution_engine_9k.pt'),
    '700k_strong': ('program_generator_700k.pt', 'execution_engine_700k_strong.pt'),
    'lstm': 'lstm.pt',
    'cnn_lstm': 'cnn_lstm.pt',
    'cnn_lstm_sa': 'cnn_lstm_sa.pt',
    'cnn_lstm_sa_mlp': 'cnn_lstm_sa_mlp.pt'
}

def main(args):
  print()
  models_path = Path(args.models_dir)
  results = {}
  for name, model_name in tqdm(models.items()):
    model = None
    vocab_path = None
    if type(model_name) is tuple:
      print('Loading program generator from ', model_name[0])
      program_generator, _ = utils.load_program_generator(models_path / model_name[0])
      print('Loading execution engine from ', model_name[1])
      execution_engine, _ = utils.load_execution_engine(models_path / model_name[1], verbose=False)
      if args.vocab_json is not None:
        new_vocab = utils.load_vocab(args.vocab_json)
        program_generator.expand_encoder_vocab(new_vocab['question_token_to_idx'])
      model = (program_generator, execution_engine)
      vocab_path = models_path / model_name[0]
    else:
      print('Loading baseline model from ', model_name)
      model, _ = utils.load_baseline(models_path / model_name)
      vocab_path = models_path / model_name
      if args.vocab_json is not None:
        new_vocab = utils.load_vocab(args.vocab_json)
        model.rnn.expand_vocab(new_vocab['question_token_to_idx'])


    result = run_raw_images(args, model, vocab_path)
    results[name] = result

  output_path = Path(args.output_dir)
  output_path.mkdir(exist_ok=True)
#   with open(output_path / 'question.txt', 'w') as f:
#     f.write(args.question)
  pd.DataFrame(results).to_csv(output_path / 'results.csv')


def load_vocab(path):
  return utils.load_cpu(path)['vocab']

def preprocess_questions(questions, vocab):
  print('Preprocessing questions')
  results = []
  for question in questions:
    question_tokens = tokenize(question,
                            punct_to_keep=[';', ','],
                            punct_to_remove=['?', '.'])
    question_encoded = encode(question_tokens,
                            vocab['question_token_to_idx'],
                            allow_unk=True)
    question_encoded = torch.LongTensor(question_encoded).view(1, -1)
    question_encoded = question_encoded.type(torch.FloatTensor).long()
    question_var = Variable(question_encoded)
    question_var.requires_grad = False
    results.append(question_var)
  results = torch.vstack(results)
  return results

def run_raw_images(args, model, vocab_path):
  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor

  # Build the CNN to use for feature extraction
  print('Loading CNN for feature extraction')
  cnn = build_cnn(args, dtype)

  # Load and preprocess the image
  img_size = (args.image_height, args.image_width)
  transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.224))])
  dataset = ClevrDataset(root_dir=args.root_dir, 
                         image_h5_filename=args.image_h5,
                         transform=transform)
  vocab = load_vocab(vocab_path)
  if args.question != None:
      questions = preprocess_questions([args.question] * len(dataset), vocab)
  else:
      assert args.questions_file != None
      question_df = pd.read_csv(args.questions_file)
      questions = preprocess_questions(question_df.values.squeeze().tolist(), vocab)
  dataset.questions = questions
  dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True
  )

  # Tokenize the question

  predicted = []
  for imgs, question_var in tqdm(dataloader):
    # encode the question

    
    # Use CNN to extract features for the image
    question_var = question_var.type(dtype).long()
    img_var = Variable(imgs.type(dtype))
    img_var.requires_grad = False
    feats_var = cnn(img_var)

    # Run the model
    # print('Running the model\n')
    scores = None
    predicted_program = None
    if type(model) is tuple:
      program_generator, execution_engine = model
      program_generator.type(dtype)
      execution_engine.type(dtype)
      predicted_programs = []
      predicted_programs = program_generator.reinforce_sample(
                                question_var,
                                temperature=args.temperature,
                                argmax=(args.sample_argmax == 1))
      scores = execution_engine(feats_var, predicted_programs)
    else:
      model.type(dtype)
      scores = model(question_var, feats_var)

    # Print results
    _, predicted_answer_idx = scores.data.cpu().max(dim=1)
    predicted_answer = [vocab['answer_idx_to_token'][i.item()] for i in predicted_answer_idx]
    predicted.extend(list(predicted_answer))
  return predicted


def build_cnn(args, dtype):
  if not hasattr(torchvision.models, args.cnn_model):
    raise ValueError('Invalid model "%s"' % args.cnn_model)
  if not 'resnet' in args.cnn_model:
    raise ValueError('Feature extraction only supports ResNets')
  whole_cnn = getattr(torchvision.models, args.cnn_model)(pretrained=True)
  layers = [
    whole_cnn.conv1,
    whole_cnn.bn1,
    whole_cnn.relu,
    whole_cnn.maxpool,
  ]
  for i in range(args.cnn_model_stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(whole_cnn, name))
  cnn = torch.nn.Sequential(*layers)
  cnn.type(dtype)
  cnn.eval()
  return cnn

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
