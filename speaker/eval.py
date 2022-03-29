import torch

from utils import *

def evaluate(encoder, decoder, vocab, batch, img_features, device="cuda"):
    """
        For mini-batch of batch_size=1 only.
    """
    EOS_token = "<<stop>>"

    assert(type(batch) == list and len(batch)==1)
    assert(type(img_features) == list and len(img_features)==1)

    # Set model to evaluation mode
    encoder.eval()
    decoder.eval()

    # Do not store gradients 
    with torch.no_grad():
        actions = [torch.tensor(vocab['action_low'].word2index([a['discrete_action']['action'] \
        for a in batch[i]['plan']['low_actions']]) + [2]).to(device) for i in range(1)]

        # Get instructions as a torch.tensor shape=(batch_size, max instruction length)
        # instructions indexed to vocab['word']
        batched_gold_instructions, instr_seq_len = get_batched_instructions(batch, device)
        # img_features will be a list [ torch.tensor's of shape(batch_size, 7*7, 512) ] 
        batched_img_features, img_seq_len = dynamic_batching(img_features, device)  
        # actions will be a list [ torch.tensor's of shape(batch_size, action_embedding_size) ]
        batched_actions, actions_seq_len  = dynamic_batching(actions, device)      
                        
        #print('eval')
        #print("batched_gold_instructions[0].shape{}".format(batched_gold_instructions[0].shape))
        #print("batched_img_features[0].shape {}".format(batched_img_features[0].shape))
        #print("batched_actions[0].shape {}".format(batched_actions[0].shape))

        # Encode all time steps of the input
        ctx, decoder_init, c = encoder.forward(batched_actions, batched_img_features, img_seq_len)
        
        # Mask context according to sequence length
        # of each element of the mini-batch
        decoder_input  = torch.full((1, 1), vocab['word'].word2index("<<seg>>")).to(device)

        # Store the words decoded
        decoded_words  = []

        decoder_hidden = decoder_init
        
        mask = time_mask(ctx, actions_seq_len)

        for t in range(batched_gold_instructions.shape[1]):
            # decoder_hidden - Hidden (batch_size, hidden_dim)
            # c_1   - CEC (batch_size, hidden_dim)
            # alpha - Attention (batch_size, hidden_dim)
            # logit - (batch_size, word_vocab_size)
            decoder_hidden, c, alpha, logit = decoder.forward(decoder_input, decoder_hidden, c, ctx, ctx_mask=None)
            topv, topi = torch.topk(logit, 1)

            # Found EOS token - stop decoding
            if topi.item() == vocab['word'].word2index(EOS_token):
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(vocab['word'].index2word(topi.item()))

            # Next decoder input is the previous word decoded
            decoder_input = topi.detach().view(1,1)     

        return decoded_words, vocab['word'].index2word(batched_gold_instructions.tolist())

