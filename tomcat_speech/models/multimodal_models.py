# the models used for multimodal, multitask classification
import sys

import torch.nn as nn

from tomcat_speech.models.audio_model_bases import AcousticOnlyModel, SpecCNNBase, SpecOnlyCNN
from tomcat_speech.models.model_prediction_layers import PredictionLayer, TextPlusPredictionLayer, AcousticPlusPredictionLayer
from tomcat_speech.models.multimodal_model_bases import IntermediateFusionMultimodalModel, EarlyFusionMultimodalModel, LateFusionMultimodalModel, MultimodalBaseDuplicateInput
from tomcat_speech.models.text_model_bases import TextOnlyModel


class MultitaskModel(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(
        self,
        params,
        num_embeddings=None,
        pretrained_embeddings=None,
        multi_dataset=True,
        use_distilbert=False
    ):
        super(MultitaskModel, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # let the network know if it's only using text features
        self.text_only = params.text_only
        self.audio_only = params.audio_only

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        if params.audio_only is True:
            self.base = AcousticOnlyModel(params)
        elif params.text_only is True:
            self.base = TextOnlyModel(
                params, num_embeddings, pretrained_embeddings, use_distilbert
            )
        elif params.spec_only is True:
            self.base = SpecOnlyCNN(params)
        else:
            if params.fusion_type.lower() == "early":
                self.base = EarlyFusionMultimodalModel(
                    params, num_embeddings, pretrained_embeddings, use_distilbert
                )
            elif params.fusion_type.lower() == "late":
                self.base = LateFusionMultimodalModel(
                    params, num_embeddings, pretrained_embeddings, use_distilbert
                )
            else:
                self.base = IntermediateFusionMultimodalModel(
                    params, num_embeddings, pretrained_embeddings, use_distilbert
                )
            # self.base = TextOnlyRNN(
            #     params, num_embeddings, pretrained_embeddings
            # )

        # set output layers
        self.task_0_predictor = PredictionLayer(params, params.output_0_dim)
        self.task_1_predictor = PredictionLayer(params, params.output_1_dim)
        self.task_2_predictor = PredictionLayer(params, params.output_2_dim)
        self.task_3_predictor = PredictionLayer(params, params.output_3_dim)
        self.task_4_predictor = PredictionLayer(params, params.output_4_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        spec_input=None,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0,
        get_prob_dist=False,
        return_penultimate_layer=False,
        save_encoded_data=False
    ):
        # NOTE: if return_penultimate_layer, forward returns a (preds, penult) double

        # call forward on base model
        if self.text_only:
            final_base_layer = self.base(
                text_input,
                speaker_input=speaker_input,
                length_input=length_input,
                gender_input=gender_input,
            )
        elif self.audio_only:
            final_base_layer = self.base(
                acoustic_input,
                speaker_input=speaker_input,
                acoustic_len_input=acoustic_len_input,
                gender_input=gender_input
            )
        else:
            final_base_layer = self.base(
                acoustic_input,
                text_input,
                spec_input=spec_input,
                speaker_input=speaker_input,
                length_input=length_input,
                acoustic_len_input=acoustic_len_input,
                gender_input=gender_input,
            )

        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None
        task_4_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_1_out = self.task_1_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_2_out = self.task_2_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_3_out = self.task_3_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_4_out = self.task_3_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 1:
                task_1_out = self.task_1_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 2:
                task_2_out = self.task_2_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 3:
                task_3_out = self.task_3_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 4:
                task_4_out = self.task_4_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            else:
                sys.exit(f"Task {task_num} not defined")

        if save_encoded_data:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out, final_base_layer
        else:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out


class MultitaskAcousticShared(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(
        self,
        params,
        num_embeddings=None,
        pretrained_embeddings=None,
        multi_dataset=True,
        use_distilbert=False
    ):
        super(MultitaskAcousticShared, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.acoustic_base = AcousticOnlyModel(params)

        # set output layers
        self.task_0_predictor = TextPlusPredictionLayer(
            params,
            params.output_0_dim,
            num_embeddings=num_embeddings,
            pretrained_embeddings=pretrained_embeddings,
            use_distilbert=use_distilbert
        )
        self.task_1_predictor = TextPlusPredictionLayer(
            params,
            params.output_1_dim,
            num_embeddings=num_embeddings,
            pretrained_embeddings=pretrained_embeddings,
            use_distilbert=use_distilbert
        )
        self.task_2_predictor = TextPlusPredictionLayer(
            params,
            params.output_2_dim,
            num_embeddings=num_embeddings,
            pretrained_embeddings=pretrained_embeddings,
            use_distilbert=use_distilbert
        )
        self.task_3_predictor = TextPlusPredictionLayer(
            params,
            params.output_3_dim,
            num_embeddings=num_embeddings,
            pretrained_embeddings=pretrained_embeddings,
            use_distilbert=use_distilbert
        )
        self.task_4_predictor = TextPlusPredictionLayer(
            params,
            params.output_4_dim,
            num_embeddings=num_embeddings,
            pretrained_embeddings=pretrained_embeddings,
            use_distilbert=use_distilbert
        )

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0,
        get_prob_dist=False,
        return_penultimate_layer=False,
        save_encoded_data=False
    ):
        # call forward on base model
        final_base_layer = self.acoustic_base(
            acoustic_input, acoustic_len_input=acoustic_len_input,
        )

        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None
        task_4_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(
                final_base_layer, text_input, speaker_input, length_input, gender_input, get_prob_dist, return_penultimate_layer
            )
            task_1_out = self.task_1_predictor(
                final_base_layer, text_input, speaker_input, length_input, gender_input, get_prob_dist, return_penultimate_layer
            )
            task_2_out = self.task_2_predictor(
                final_base_layer, text_input, speaker_input, length_input, gender_input, get_prob_dist, return_penultimate_layer
            )
            task_3_out = self.task_3_predictor(
                final_base_layer, text_input, speaker_input, length_input, gender_input, get_prob_dist, return_penultimate_layer
            )
            task_4_out = self.task_4_predictor(
                final_base_layer, text_input, speaker_input, length_input, gender_input, get_prob_dist, return_penultimate_layer
            )
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(
                    final_base_layer,
                    text_input,
                    speaker_input,
                    length_input,
                    gender_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 1:
                task_1_out = self.task_1_predictor(
                    final_base_layer,
                    text_input,
                    speaker_input,
                    length_input,
                    gender_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 2:
                task_2_out = self.task_2_predictor(
                    final_base_layer,
                    text_input,
                    speaker_input,
                    length_input,
                    gender_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 3:
                task_3_out = self.task_3_predictor(
                    final_base_layer,
                    text_input,
                    speaker_input,
                    length_input,
                    gender_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 4:
                task_4_out = self.task_4_predictor(
                    final_base_layer,
                    text_input,
                    speaker_input,
                    length_input,
                    gender_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            else:
                sys.exit(f"Task {task_num} not defined")

        if save_encoded_data:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out, final_base_layer
        else:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out


class MultitaskTextShared(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(
        self,
        params,
        num_embeddings=None,
        pretrained_embeddings=None,
        multi_dataset=True,
        use_distilbert=False
    ):
        super(MultitaskTextShared, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.text_base = TextOnlyModel(params,
                                           num_embeddings=num_embeddings,
                                           pretrained_embeddings=pretrained_embeddings,
                                           use_distilbert=use_distilbert)

        # set output layers
        self.task_0_predictor = AcousticPlusPredictionLayer(
            params,
            params.output_0_dim
        )
        self.task_1_predictor = AcousticPlusPredictionLayer(
            params,
            params.output_1_dim
        )
        self.task_2_predictor = AcousticPlusPredictionLayer(
            params,
            params.output_2_dim
        )
        self.task_3_predictor = AcousticPlusPredictionLayer(
            params,
            params.output_3_dim
        )
        self.task_4_predictor = AcousticPlusPredictionLayer(
            params,
            params.output_4_dim
        )

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0,
        get_prob_dist=False,
        return_penultimate_layer=False,
        save_encoded_data=False
    ):
        # call forward on base model
        final_base_layer = self.text_base(
            text_input, length_input=length_input,
        )

        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None
        task_4_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(
                acoustic_input, final_base_layer, acoustic_len_input, get_prob_dist, return_penultimate_layer
            )
            task_1_out = self.task_1_predictor(
                acoustic_input, final_base_layer, acoustic_len_input, get_prob_dist, return_penultimate_layer
            )
            task_2_out = self.task_2_predictor(
                acoustic_input, final_base_layer, acoustic_len_input, get_prob_dist, return_penultimate_layer
            )
            task_3_out = self.task_3_predictor(
                acoustic_input, final_base_layer, acoustic_len_input, get_prob_dist, return_penultimate_layer
            )
            task_4_out = self.task_4_predictor(
                acoustic_input, final_base_layer, acoustic_len_input, get_prob_dist, return_penultimate_layer
            )
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(
                    acoustic_input,
                    final_base_layer,
                    acoustic_len_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 1:
                task_1_out = self.task_1_predictor(
                    acoustic_input,
                    final_base_layer,
                    acoustic_len_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 2:
                task_2_out = self.task_2_predictor(
                    acoustic_input,
                    final_base_layer,
                    acoustic_len_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 3:
                task_3_out = self.task_3_predictor(
                    acoustic_input,
                    final_base_layer,
                    acoustic_len_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            elif task_num == 4:
                task_4_out = self.task_4_predictor(
                    acoustic_input,
                    final_base_layer,
                    acoustic_len_input,
                    get_prob_dist,
                    return_penultimate_layer
                )
            else:
                sys.exit(f"Task {task_num} not defined")

        if save_encoded_data:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out, final_base_layer
        else:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out


class MultitaskDuplicateInputModel(nn.Module):
    """
    A multimodal model that duplicates input modality tensors
    Inspired by:
    Frustratingly easy domain adaptation
    """

    def __init__(
        self,
        params,
        num_embeddings=None,
        pretrained_embeddings=None,
        multi_dataset=True,
        num_tasks=5,
        use_distilbert=False
    ):
        super(MultitaskDuplicateInputModel, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = MultimodalBaseDuplicateInput(
            params, num_embeddings, pretrained_embeddings, num_tasks, use_distilbert
        )

        # set output layers
        self.task_0_predictor = PredictionLayer(params, params.output_0_dim)
        self.task_1_predictor = PredictionLayer(params, params.output_1_dim)
        self.task_2_predictor = PredictionLayer(params, params.output_2_dim)
        self.task_3_predictor = PredictionLayer(params, params.output_3_dim)
        self.task_4_predictor = PredictionLayer(params, params.output_4_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0,
        get_prob_dist=False,
        return_penultimate_layer=False,
        save_encoded_data=False
    ):
        # call forward on base model
        final_base_layer = self.base(
            acoustic_input,
            text_input,
            length_input=length_input,
            gender_input=gender_input,
            task_num=task_num,
        )

        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None
        task_4_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_1_out = self.task_1_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_2_out = self.task_2_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_3_out = self.task_3_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_4_out = self.task_4_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 1:
                task_1_out = self.task_1_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 2:
                task_2_out = self.task_2_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 3:
                task_3_out = self.task_3_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 4:
                task_4_out = self.task_4_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            else:
                sys.exit(f"Task {task_num} not defined")

        if save_encoded_data:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out, final_base_layer
        else:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out
