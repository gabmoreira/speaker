import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F


class VisualSoftDotAttention(nn.Module):
    """
        Visual Dot Attention Layer. 
        As in Show, Attend and Tell paper
    """
    def __init__(self, h_dim, v_dim, dot_dim=256):

        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):
        """
            Propagate h through the network.
            h: batch x hidden_dim
            visual_context: (batch, #image_regions-7x7, #features_per_region typically 512)
        """
        #print('VisualSoftDotAttention')
        target  = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim

        #print("\tHidden h: {}".format(h.shape))
        #print("\tVisual visual_context: {}".format(visual_context.shape))

        #print("\tHidden projected h: {}".format(target.shape))
        #print("\tVisual context projected: {}".format(context.shape))

        # Dot products between h and all the feature vectors in all image regions
        attn  = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn  = self.softmax(attn)

        #print("\tAttention map: {}".format(attn.shape))

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        # Weigh the visual vector by the attention
        weighted_context = torch.bmm(attn3, visual_context).squeeze(1)  # batch x v_dim

        #print("\tWeighted visual context map: {}".format(weighted_context.shape))

        return weighted_context, attn



class SoftDotAttention(nn.Module):
    """
        Soft Dot Attention.
        Ref: http://www.aclweb.org/anthology/D15-1166
        Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """
            Initialize layer.
        """
        super(SoftDotAttention, self).__init__()
        self.linear_in  = nn.Linear(dim, dim, bias=False)
        self.softmax    = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh       = nn.Tanh()

    def forward(self, h, context, mask=None):
        """
            Propagate h through the network.
            h: (batch_size, dim)
            context: (batch_size, seq_len, dim)
            mask: batch x seq_len indices to be masked
        """
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention as (batch, seq_len)
        attn = torch.bmm(context, target).squeeze(2)  # 
        if mask is not None:
            # -Inf masking prior to the softmax
            # Fill with -inf where mask=1
            attn.data.masked_fill_(mask, -float('inf'))

        attn  = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn



class ContextOnlySoftDotAttention(nn.Module):
    """
        Like SoftDot, but don't concatenat h or perform the non-linearity transform
        Ref: http://www.aclweb.org/anthology/D15-1166
        Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim, context_dim=None):
        '''Initialize layer.'''
        super(ContextOnlySoftDotAttention, self).__init__()
        if context_dim is None:
            context_dim = dim
        self.linear_in = nn.Linear(dim, context_dim, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        """
            Propagate h through the network.
            h: batch x dim
            context: batch x seq_len x dim
            mask: batch x seq_len indices to be masked
        """
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn



class EncoderLSTM(nn.Module):
    def __init__(self, action_vocab_size, action_embedding_size, visual_embedding_size,  hidden_size, dropout_ratio=0.1, device="cuda"):
        """
        """
        super(EncoderLSTM, self).__init__()

        self.action_embedding_size = action_embedding_size
        self.word_embedding_size   = visual_embedding_size
        self.hidden_size           = hidden_size

        self.action_embedding       = nn.Embedding(action_vocab_size, action_embedding_size)
        self.drop                   = nn.Dropout(p=dropout_ratio)
        self.visual_attention_layer = VisualSoftDotAttention(hidden_size, visual_embedding_size)
        self.lstm                   = nn.LSTMCell(action_embedding_size + visual_embedding_size, hidden_size)
        self.encoder2decoder        = nn.Linear(hidden_size, hidden_size)

        self.device = device

    def init_state(self, batch_size):
        """
            Initialize to zero cell states and hidden states.
        """
        h0 = Variable(torch.zeros(batch_size, self.hidden_size, device=self.device), requires_grad=False) # Hidden state
        c0 = Variable(torch.zeros(batch_size, self.hidden_size, device=self.device), requires_grad=False) # Constant error carroussel
        return h0, c0


    def _forward_one_step(self, h_0, c_0, action, visual_features):
        """
            Forward one time-step
        """
        action_embedding = self.action_embedding(action)
        feature, _       = self.visual_attention_layer(h_0, visual_features)
        #print("Concatenating action embedding {} and attended visual {}\n".format(action_embedding.shape, feature.shape))
        concat_input     = torch.cat((action_embedding, feature), 1)

        drop = self.drop(concat_input)

        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        return h_1, c_1


    def forward(self, batched_actions, batches_visual_features, seq_len):
        """
            Forward over entire time span

            Parameters:
            ___________
            batched_actions (list): List of len=maximum sequence length.
            [torch.tensor; shape=(batch_size, 1)]

            batches_visual_features (list): List of len=maximum sequence length.
            [torch.tensor; shape=(batch_size, *, visual_embedding_size)]

            Returns:
            ________
            ctx (torch.tensor) - context tensor; shape=(batch_size, max sequence length, hidden_dim)
            decoder_init (torch.tensor) - final hidden state; shape=(batch_size, hidden_dim)
            c (torch.tensor) - CEC; shape=(batch_size, hidden_dim)
        """   
        assert isinstance(batched_actions, list)
        assert isinstance(batches_visual_features, list)
        assert len(batched_actions) == len(batches_visual_features)

        batch_size = batches_visual_features[0].shape[0]

        h, c = self.init_state(batch_size)
        h_list = []

        # Iterate through time
        for t, (action_embedding, visual_features) in enumerate(zip(batched_actions, batches_visual_features)):
            #print("Forward t={}".format(t))
            h, c = self._forward_one_step(h, c, action_embedding, visual_features)
            h_list.append(h)

        decoder_init = nn.Tanh()(self.encoder2decoder(h))

        ctx = torch.stack(h_list, dim=1)  # (batch, seq_len, hidden_size)
        ctx = self.drop(ctx)

        return ctx, decoder_init, c  # (batch, hidden_size)




class DecoderLSTM(nn.Module):

    def __init__(self, vocab_size, vocab_embedding_size, hidden_size, dropout_ratio=0.1, use_input_att_feed=False):
        super(DecoderLSTM, self).__init__()

        self.use_input_att_feed   = use_input_att_feed
        self.vocab_size           = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size          = hidden_size

        self.embedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.drop      = nn.Dropout(p=dropout_ratio)

        if self.use_input_att_feed:
            #print('Using input attention feed in SpeakerDecoderLSTM')
            self.lstm            = nn.LSTMCell(vocab_embedding_size + hidden_size, hidden_size)
            self.attention_layer = ContextOnlySoftDotAttention(hidden_size)
            self.output_l1       = nn.Linear(hidden_size*2, hidden_size)
            self.tanh            = nn.Tanh()
        else:
            self.lstm            = nn.LSTMCell(vocab_embedding_size, hidden_size)
            self.attention_layer = SoftDotAttention(hidden_size)

        self.decoder2word = nn.Linear(hidden_size, vocab_size)


    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        """
            Takes a single step in the decoder LSTM (allowing sampling).
            action: batch x 1
            feature: batch x feature_size
            h_0: batch x hidden_size
            c_0: batch x hidden_size
            ctx: batch x seq_len x dim
            ctx_mask: batch x seq_len - indices to be masked
        """
        #print("Decoder previous word {}".format(previous_word.shape))
        word_embeds = self.embedding(previous_word)
        #print("Decoder embedded previous word {}".format(word_embeds.shape))
        word_embeds = word_embeds.squeeze(1)  # (batch, embedding_size)

        word_embeds_drop = self.drop(word_embeds)

        if self.use_input_att_feed:
            h_tilde, alpha = self.attention_layer(self.drop(h_0), ctx, ctx_mask)
            concat_input = torch.cat((word_embeds_drop, self.drop(h_tilde)), 1)
            h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
            x = torch.cat((h_1, h_tilde), 1)
            x = self.drop(x)
            x = self.output_l1(x)
            x = self.tanh(x)
            logit = self.decoder2word(x)
            logit = F.log_softmax(logit, dim=1)
        else:
            h_1, c_1       = self.lstm(word_embeds_drop, (h_0, c_0))
            h_1_drop       = self.drop(h_1)
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
            logit          = self.decoder2word(h_tilde)
            logit          = F.log_softmax(logit, dim=1)

        return h_1, c_1, alpha, logit
