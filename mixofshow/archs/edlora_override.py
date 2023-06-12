import torch
from diffusers.models.cross_attention import CrossAttention


def remove_unet_attention_forward(unet):

    def change_forward(unet):  # omit proceesor in new diffusers
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'CrossAttention' and name == 'attn2':
                bound_method = CrossAttention.forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer)

    change_forward(unet)


def revise_unet_attention_forward(unet):

    def get_new_forward(cross_attention_idx):

        def new_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            else:
                encoder_hidden_states = encoder_hidden_states[:, cross_attention_idx, ...]

            if self.cross_attention_norm:
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states

        return new_forward

    def change_forward(unet, cross_attention_idx):  # omit proceesor in new diffusers
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'CrossAttention' and name == 'attn2':
                bound_method = get_new_forward(cross_attention_idx).__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
                cross_attention_idx += 1
            else:
                cross_attention_idx = change_forward(layer, cross_attention_idx)
        return cross_attention_idx

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx)
    change_forward(unet.up_blocks, cross_attention_idx)
