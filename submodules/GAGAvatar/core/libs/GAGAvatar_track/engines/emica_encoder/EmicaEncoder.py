#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modified based on code from  Radek Danecek (Max-Planck-Gesellschaft zur Förderung).

import os
import torch 
from .DecaEncoder import DecaResnetEncoder
from .MicaEncoder import MicaArcfaceEncoder

class EmicaEncoder(torch.nn.Module): 
    def __init__(self, device='cpu'):
        super().__init__()
        self._device = device

    def _init_models(self, ):
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.path.join('assets/emica/EMICA-CVT_flame2020_notexture.pt')
        assert os.path.exists(_model_path), f"Model not found: {_model_path}."
        ckpt = torch.load(_model_path, map_location='cpu', weights_only=True)
        self.mica_encoder = MicaEncoder()
        self.deca_encoder = DecaEncoder(outsize=86)
        self.expression_encoder = DecaEncoder(outsize=100)
        self.load_state_dict(ckpt)
        self.mica_encoder.to(self._device).eval()
        self.deca_encoder.to(self._device).eval()
        self.expression_encoder.to(self._device).eval()

    def forward(self, inp_images):
        if not hasattr(self, 'mica_encoder'):
            self._init_models()
        flame_results = self.deca_encoder.encode(inp_images['warped_image'].float())
        mica_shape, _ = self.mica_encoder.encode(inp_images['mica_image'].float())
        flame_results['shapecode'] = mica_shape
        exp_code = self.expression_encoder.encode(inp_images['warped_image'].float())
        flame_results['expcode'] = exp_code['expcode']
        # shape, expression, jaw, global, cam, light
        return flame_results


class DecaEncoder(torch.nn.Module):
    def __init__(self, outsize=100):
        super().__init__()
        self.encoder = DecaResnetEncoder(outsize=outsize, last_op=None)
        self.encoder.requires_grad_(False)
        if outsize == 100:
            self._prediction_code_dict = {
                'expcode': 100
            }
        elif outsize == 86:
            self._prediction_code_dict = {
                'texcode': 50, 'jawpose': 3, 'globalpose': 3, 'cam': 3, 'lightcode': 27
            }
        else:
            raise ValueError(f"Invalid outsize: {outsize}")

    @torch.no_grad()
    def encode(self, deca_images):
        code_vec = self.encoder(deca_images, output_features=False)
        results = self._decompose_code(code_vec)
        return results
    
    def _decompose_code(self, code):
        '''
        Decompose the code into the different components based on the prediction_code_dict
        '''
        results = {}
        start = 0
        for key, dim in self._prediction_code_dict.items():
            subcode = code[..., start:start + dim]
            if key == 'light':
                subcode = subcode.reshape(subcode.shape[0], 9, 3)
            results[key] = subcode
            start = start + dim
        return results


class MicaEncoder(torch.nn.Module): 
    def __init__(self, ):
        super().__init__()
        self.E_mica = MicaArcfaceEncoder()

    @torch.no_grad()
    def encode(self, mica_image):
        mica_encoding = self.E_mica.encode(mica_image) 
        mica_shapecode, identity_code = self.E_mica.decode(mica_encoding)
        return mica_shapecode, identity_code

