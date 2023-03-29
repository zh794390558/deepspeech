# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantzation U2 model."""
import paddle
from kaldiio import ReadHelper
from paddle.quantization import PTQ
from paddle.quantization import QuantConfig
from paddleslim.quant.observers import AVGObserver
from paddleslim.quant.observers import EMDObserver
from paddleslim.quant.observers import HistObserver
from paddleslim.quant.observers import KLObserver
from paddleslim.quant.observers import MSEChannelWiseWeightObserver
from paddleslim.quant.observers import MSEObserver
from paddleslim.quant.observers.abs_max_weight import AbsMaxChannelWiseWeightObserver
from paddleslim.quant.observers.uniform import UniformObserver
from yacs.config import CfgNode

from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.models.u2 import U2Model
from paddlespeech.s2t.modules import align
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig
logger = Log(__name__).getlog()


class U2Infer():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.audio_scp = args.audio_scp

        self.preprocess_conf = config.preprocess_config
        self.preprocess_args = {"train": False}
        self.preprocessing = Transformation(self.preprocess_conf)
        self.text_feature = TextFeaturizer(
            unit_type=config.unit_type,
            vocab=config.vocab_filepath,
            spm_model_prefix=config.spm_model_prefix)

        paddle.set_device('gpu' if self.args.ngpu > 0 else 'cpu')

        # model
        model_conf = config
        with UpdateConfig(model_conf):
            model_conf.input_dim = config.feat_dim
            model_conf.output_dim = self.text_feature.vocab_size
        model = U2Model.from_config(model_conf)
        self.model = model
        self.model.eval()

        # ptq
        self.q_config = QuantConfig(activation=None, weight=None)

        if args.act_bits == 32:
            act_observer = None
        elif args.act_abserver == 'avg':
            act_observer = AVGObserver(quant_bits=args.act_bits)
        elif args.act_abserver == 'hist':
            act_observer = HistObserver(
                percent=args.percent, quant_bits=args.act_bits)
        elif args.act_abserver == 'mse':
            act_observer = MSEObserver(quant_bits=args.act_bits)
        elif args.act_abserver == 'kl':
            act_observer = KLObserver(quant_bits=args.act_bits)
        elif args.act_abserver == 'emd':
            act_observer = EMDObserver(quant_bits=args.act_bits)
        elif args.act_abserver == "uniform":
            act_observer = UniformObserver(quant_bits=args.act_bits)
        else:
            raise ValueError('Unknown activation strategy: %s' %
                             args.act_abserver)

        if args.weight_bits == 32:
            weight_observer = None
        elif args.weight_abserver == 'mse':
            weight_observer = MSEObserver(quant_bits=args.weight_bits)
        elif args.weight_abserver == 'mse-channelwise':
            weight_observer = MSEChannelWiseWeightObserver(
                quant_bits=args.weight_bits)
        elif args.weight_abserver == 'absmax-channelwise':
            weight_observer = AbsMaxChannelWiseWeightObserver(
                quant_bits=args.weight_bits)
        else:
            raise ValueError("Unknown weight strategy: %s" %
                             args.weight_abserver)

        print('act_observer', act_observer, 'weight_observer', weight_observer)
        self.q_config.add_type_config(
            [align.Linear, align.Conv2D],
            activation=act_observer,
            weight=weight_observer)
        # self.q_config.add_qat_layer_mapping(align.Linear, paddle.nn.quant.quant_layers.QuantizedLinear)
        # self.q_config.add_qat_layer_mapping(align.Conv2D, paddle.nn.quant.quant_layers.QuantizedConv2D)

        self.ptq = PTQ(self.q_config)
        self.model = self.ptq.quantize(model, inplace=False)

        # load model
        params_path = self.args.checkpoint_path + ".pdparams"
        model_dict = paddle.load(params_path)
        self.model.set_state_dict(model_dict)

    def run(self):
        cnt = 0
        with ReadHelper(f"scp:{self.audio_scp}") as reader:
            for key, (rate, audio) in reader:
                assert rate == 16000
                cnt += 1
                if cnt > args.num_utts:
                    break

                with paddle.no_grad():
                    logger.info(f"audio shape: {audio.shape}")

                    # fbank
                    feat = self.preprocessing(audio, **self.preprocess_args)
                    logger.info(f"feat shape: {feat.shape}")

                    ilen = paddle.to_tensor(feat.shape[0])
                    xs = paddle.to_tensor(feat, dtype='float32').unsqueeze(0)
                    decode_config = self.config.decode
                    logger.info(f"decode cfg: {decode_config}")
                    result_transcripts = self.model.decode(
                        xs,
                        ilen,
                        text_feature=self.text_feature,
                        decoding_method=decode_config.decoding_method,
                        beam_size=decode_config.beam_size,
                        ctc_weight=decode_config.ctc_weight,
                        decoding_chunk_size=decode_config.decoding_chunk_size,
                        num_decoding_left_chunks=decode_config.
                        num_decoding_left_chunks,
                        simulate_streaming=decode_config.simulate_streaming,
                        reverse_weight=decode_config.reverse_weight)
                    rsl = result_transcripts[0][0]
                    utt = key
                    logger.info(f"hyp: {utt} {rsl}")
                    # print(self.model)
                    # print(self.model.forward_encoder_chunk)

        logger.info("-------------start quant ----------------------")
        batch_size = 1
        feat_dim = 80
        model_size = 512
        num_left_chunks = -1
        reverse_weight = 0.3
        logger.info(
            f"U2 Export Model Params: batch_size {batch_size}, feat_dim {feat_dim}, model_size {model_size}, num_left_chunks {num_left_chunks}, reverse_weight {reverse_weight}"
        )

        # ######################## self.model.forward_encoder_chunk ############
        # input_spec = [
        #     # (T,), int16
        #     paddle.static.InputSpec(shape=[None], dtype='int16'),
        # ]
        # self.model.forward_feature = paddle.jit.to_static(
        #     self.model.forward_feature, input_spec=input_spec)

        ######################### self.model.forward_encoder_chunk ############
        input_spec = [
            # xs, (B, T, D)
            paddle.static.InputSpec(
                shape=[batch_size, None, feat_dim], dtype='float32'),
            # offset, int, but need be tensor
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # required_cache_size, int
            num_left_chunks,
            # att_cache
            paddle.static.InputSpec(
                shape=[None, None, None, None], dtype='float32'),
            # cnn_cache
            paddle.static.InputSpec(
                shape=[None, None, None, None], dtype='float32')
        ]
        self.model.forward_encoder_chunk = paddle.jit.to_static(
            self.model.forward_encoder_chunk, input_spec=input_spec)

        ######################### self.model.ctc_activation ########################
        input_spec = [
            # encoder_out, (B,T,D)
            paddle.static.InputSpec(
                shape=[batch_size, None, model_size], dtype='float32')
        ]
        self.model.ctc_activation = paddle.jit.to_static(
            self.model.ctc_activation, input_spec=input_spec)

        ######################### self.model.forward_attention_decoder ########################
        input_spec = [
            # hyps, (B, U)
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            # hyps_lens, (B,)
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            # encoder_out, (B,T,D)
            paddle.static.InputSpec(
                shape=[batch_size, None, model_size], dtype='float32'),
            reverse_weight
        ]
        self.model.forward_attention_decoder = paddle.jit.to_static(
            self.model.forward_attention_decoder, input_spec=input_spec)
        ################################################################################

        # convert to onnx quant format
        self.ptq.convert(self.model, inplace=True)

        # jit save
        logger.info(f"export save: {self.args.export_path}")
        paddle.jit.save(
            self.model,
            self.args.export_path,
            combine_params=True,
            skip_forward=True)


def main(config, args):
    U2Infer(config, args).run()


if __name__ == "__main__":
    parser = default_argument_parser()
    # save asr result to
    parser.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    parser.add_argument(
        "--audio_scp", type=str, help="path of the input audio file")
    parser.add_argument(
        "--num_utts",
        type=int,
        default=200,
        help="num utts for quant calibrition.")
    parser.add_argument(
        "--export_path",
        type=str,
        default='export.jit.quant',
        help="path of the input audio file")
    parser.add_argument(
        "--act-bits", type=int, default=8, help="activation quant bits")
    parser.add_argument(
        "--act-abserver",
        type=str,
        default="emd",
        choices=["emd", "avg", "hist", "mse", "kl", "uniform"],
        help="activation abserver type")
    parser.add_argument(
        "--weight-bits", type=int, default=8, help="activation quant bits")
    parser.add_argument(
        "--weight-abserver",
        type=str,
        default="mse-channelwise",
        choices=["mse-channelwise", "absmax-channelwise", "mse"],
        help="weight abserver type")
    # https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/quant/observers/hist.py
    parser.add_argument(
        "--hist-percent",
        type=float,
        default=0.999,
        help="HistObserver: the percentage of bins that are retained when clipping the outliers"
    )
    args = parser.parse_args()

    config = CfgNode(new_allowed=True)

    if args.config:
        config.merge_from_file(args.config)
    if args.decode_cfg:
        decode_confs = CfgNode(new_allowed=True)
        decode_confs.merge_from_file(args.decode_cfg)
        config.decode = decode_confs
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    main(config, args)
