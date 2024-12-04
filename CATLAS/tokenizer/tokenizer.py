import pandas as pd
import sentencepiece as spm
import json
import pickle

class TranscriptTokenizer():

    '''Construct a tokenizer using byte pair encoding (BPE) using sentencepiece.
    
    Args:
        transcripts_file (str):
            Path to the text file containing the transcripts.
        vocab_file (str):
            Path to the vocabulary file.
        sos_token (str, optional, ddefaults to '<s>'):
            Start of sequence token.
        eos_token (str, optional, defaults to '</s>'):
            End of sequence token.
        pad_token (str, optional, defaults to '<pad>'):
            Padding token.
        unk_token (str, optional, defaults to '<unk>'):
            Unknown token.
        -_token (str, optional, defaults to '-'):
            Token to distinguish genes.

    '''
    def __init__(self, text_file):
        self.text_file = text_file
    
    def train_sp_tokenizer(self, vocab_size):

        spm.SentencePieceTrainer.train(input=self.text_file, model_prefix='bpe',vocab_size=vocab_size, model_type='bpe')
        sp = spm.SentencePieceProcessor()
        sp_model = 'bpe.model'

        
        #Extract the vocabulary
        sp.load(sp_model)
        vocab = {}
        for id in range(sp.get_piece_size()):
            vocab[sp.id_to_piece(id)] = id
        
        vocab_file = 'vocabulary.json'
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=4)
        print(f"Vocabulary saved to {vocab_file}")

        return sp, sp_model
    

    def train_gene_name_tokenizer(self, elements):
        #elemnts = list of gene names

        #special_tokens = ['<sos>','<eos>','<unk>', '<s>', '</s>', '-', ' ']
        special_tokens = ['<sos>','<eos>','<unk>']
        # Start vocab with 1 since 0 is reserved for padding
        vocab = {token: idx+1 for idx, token in enumerate(special_tokens)}

        filtered_elements = [elem for elem in elements if not (elem.startswith('NegControlProbe') or elem.startswith('UnassignedCodeword') or elem.startswith('NegControlCodeword'))]
        start_idx = len(vocab)+1

        for token in filtered_elements:
            if token not in vocab:
                vocab[token] = start_idx
                start_idx += 1

        # Save the updated vocab to vocab.json
        with open('vocab_gene_name.json', 'w') as f:
            json.dump(vocab, f, indent=4)
        
        return vocab
    
    def tune_gene_name_tokenizer(self, elements, vocab_file):

        with open (vocab_file, 'rb') as f:
            vocab = json.load(f)
        
        filtered_elements = [elem for elem in elements if not (elem.startswith('NegControlProbe') or elem.startswith('UnassignedCodeword') or elem.startswith('NegControlCodeword'))]
        start_idx = len(vocab)+1

        for token in filtered_elements:
            if token not in vocab:
                vocab[token] = start_idx
                start_idx += 1

        # Save the updated vocab to vocab.json
        with open('vocab_gene_name.json', 'w') as f:
            json.dump(vocab, f, indent=4)
        
        return vocab


    def encode_gene_name(self, text, vocabulary, vocab_path):

        def load_vocabulary(vocab_path):
            with open(vocab_path, 'r') as file:
                vocabulary = json.load(file)
            return vocabulary
        
        def read_sentences(file_path):
            with open(file_path, 'r') as file:
                sentences = file.readlines()
            return [sentence.strip() for sentence in sentences]

        def encode_word(word, vocabulary):
            return vocabulary.get(word, vocabulary.get('<unk>'))

        def encode_sentence(sentences, vocabulary):
            encoded_sentences = []
            for sentence in sentences:
                encoded_sentence = []
                for part in sentence.split(' '):
                    encoded_tokens = [encode_word(token, vocabulary) for token in part.split('--') if token]
                    if encoded_tokens:
                        encoded_sentence.append(encoded_tokens)
                if encoded_sentence:
                    encoded_sentences.append(encoded_sentence)
            return encoded_sentences

        #vocabulary = load_vocabulary(vocab_path)
        sentences = read_sentences(text)
        encoded_sentences = encode_sentence(sentences, vocabulary)

        with open('gene_expressions_tokenized.txt', 'w') as file:
            for encoded_sentence in encoded_sentences:
                file.write(str(encoded_sentence) + '\n')
        
        return encoded_sentences

    
    def encode_sp(self, sp, sp_model, text):
        sp.load(sp_model)

        with open(text, 'r') as file:
            gene_expressions = file.readlines()

        gene_expressions_tokenized = [sp.encode(exp.strip(), out_type=int) for exp in gene_expressions]

        output_file = 'gene_expressions_tokenized.txt'
        with open (output_file, 'w') as file:
            for token in gene_expressions_tokenized:
                file.write(f'{token}\n')
        
        with open('gene_expressions_tokenized', 'wb') as file:
            pickle.dump(gene_expressions_tokenized, file)

        return gene_expressions_tokenized
    
    def decode_sp(self, sp, sp_model, encoded):
        sp.load(sp_model)
        return sp.decode_ids(encoded)
