import torch
from torch import nn

import modules
from datetime import datetime

class Embeddings(nn.Module):
    def __init__(self, embedding_dim=64):
        super(Embeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings_holiday = modules.CharacterEmbeddings(12, embedding_dim)
        self.embeddings_weather = modules.CharacterEmbeddings(11, embedding_dim)
        self.embeddings_weather_detail = modules.CharacterEmbeddings(38, embedding_dim)
        self.embeddings_month = modules.CharacterEmbeddings(12, embedding_dim)
        self.embeddings_dayofweek = modules.CharacterEmbeddings(7, embedding_dim)
        self.embeddings_hour = modules.CharacterEmbeddings(24, embedding_dim)

    def forward(self, data_dict):
        embed1 = self.embeddings_holiday.forward(data_dict['code_holiday'])
        embed2 = self.embeddings_weather.forward(data_dict['code_weather'])
        embed3 = self.embeddings_weather_detail.forward(data_dict['code_weather_detail'])
        embed4 = self.embeddings_month.forward(data_dict['code_month'])
        embed5 = self.embeddings_dayofweek.forward(data_dict['code_dayofweek'])
        embed6 = self.embeddings_hour.forward(data_dict['code_hour'])
        return torch.cat([embed1, embed2, embed3, embed4, embed5, embed6], 1)

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.linears = modules.LinearSeq(561, [1024, 128, 64, 32, 16, 6], activation_list=['relu', 'relu', 'relu', 'relu', 'relu', 'logsoftmax'])
        # self.linears = modules.LinearSeq(28, [32, 1], activation_list=['relu', None])

    def train(self, x_train, y_train, criterion, optimizer, num_epochs=1000):
        # hparams
        batch_size = 256

        # 1. set device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        print('device:', device)
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        # 2. network to device
        self.to(device)

        # 3. loop over epoch
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(num_epochs):
                start = datetime.now()
                print('---------------\nEpoch ', epoch + 1, '\n')
                epoch_loss = .0

                num_batches = x_train.size(0) // batch_size

                # 3.1 loop over batch
                batch_count = 0
                while True:
                    if batch_count < num_batches:
                        batch = x_train[batch_size * batch_count: batch_size * (batch_count + 1)]
                        labels = y_train[batch_size * batch_count: batch_size * (batch_count + 1)].squeeze()
                    else:
                        batch = x_train[batch_size * batch_count:]
                        labels = y_train[batch_size * batch_count:].squeeze()
                    # print('Epoch:', epoch+1, '/', num_epochs, 'Batch:', batch_count, '/', num_batches)
                    # 3.1.0 initialize grads
                    optimizer.zero_grad()

                    # 3.1.1 linears
                    preds = self.linears.forward(batch)

                    # 3.1.3 calc batch loss
                    loss = criterion(preds, labels)

                    # 3.1.4 calc grads
                    loss.backward(retain_graph=True)

                    # 3.1.5 update model params
                    optimizer.step()

                    # 3.1.6 add batch loss to epoch loss
                    epoch_loss += loss.item() * batch.size(0)
                    # print('epoch loss: ', epoch_loss)

                    batch_count += 1
                    if batch_count > num_batches:
                        break
                end = datetime.now()
                # 3.2 calc epoch loss
                epoch_loss /= x_train.size(0)
                print('Epoch', epoch + 1, 'average loss:', epoch_loss, 'elapsed:', (end - start).seconds + round(
                    (end - start).microseconds / 1000000, 2))

    def eval(self, x_test):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        self.to(device)
        print('device:', device)
        x_test = x_test.to(device)
        with torch.no_grad():
            preds = self.linears.forward(x_test)
            return preds.max(dim=1)[1] + 1 # returns max index + 1

# class Tacotron2(nn.Module):
#     def __init__(self, *args):
#         super(Tacotron2, self).__init__()
#         self.encoder = tacotron.modules.Encoder(*args)
#         self.attention = tacotron.modules.LocationSensitiveAttention(*args)
#         self.decoder = tacotron.modules.Decoder(*args)
#
    # def pseudo_train(self, criterion, optimizer, num_epochs=100):
    #     # 1. set device
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     print('device:', device)
    #
    #     # 2. network to device
    #     self.to(device)
    #     batch_size = 4
    #     max_input_length = 100
    #     input_character_indices = torch.randint(0, 30, [3, batch_size, max_input_length])
    #     labels = (torch.rand_like(self.decoder.spectrogram_pred),
    #               torch.rand_like(self.decoder.spectrogram_length_pred.type(torch.float32)))
    #
    #     # 3. loop over epoch
    #     with torch.autograd.set_detect_anomaly(True):
    #         for epoch in range(num_epochs):
    #             print('---------------\nEpoch ', epoch + 1, '\n')
    #             epoch_loss = .0
    #             epoch_correct = 0
    #
    #             # 3.1 loop over batch
    #             for batch in input_character_indices:
    #                 # 3.1.0 initialize grads and decoder attributes
    #                 optimizer.zero_grad()
    #                 self.decoder.reset(batch_size)
    #
    #                 # 3.1.1 encoder
    #                 encoder_output, (encoder_h_n, encoder_c_n) = self.encoder(batch)
    #                 self.attention.h = encoder_output
    #                 h_prev_1 = self.decoder.h_prev_1.clone()
    #                 stop_token_cum = self.decoder.stop_token_cum.clone()
    #
    #                 # 3.1.2 loop over decoder step
    #                 for decoder_step in range(self.decoder.max_output_time_length):
    #                     print('\n---------------------', 'decoder step: ', decoder_step + 1)
    #                     context_vector = self.attention.forward(h_prev_1, stop_token_cum)
    #                     h_prev_1, stop_token_cum = self.decoder.forward(context_vector)
    #                     if not any(
    #                             stop_token_cum):  # stop decoding if no further prediction is needed for any samples in batch
    #                         break
    #
    #                 # 3.1.3 calc batch loss
    #                 length_pred_norm = self.decoder.spectrogram_length_pred.type(
    #                     torch.float32) / self.decoder.max_output_time_length
    #                 preds = (self.decoder.spectrogram_pred, length_pred_norm)
    #                 loss = criterion(preds, labels)
    #
    #                 # 3.1.4 calc grads
    #                 loss.backward()
    #
    #                 # 3.1.5 update model params
    #                 optimizer.step()
    #
    #                 # 3.1.6 add batch loss to epoch loss
    #                 epoch_loss += loss.item() * batch.size(0)
    #
    #             # 3.2 calc epoch loss
    #             epoch_loss /= input_character_indices.size(0) * input_character_indices.size(1)

    # def train(self, dataloaders_dict, criterion, optimizer, num_epochs=100):
    #     # 1. set device
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     print('device:', device)
    #     # 2. network to device
    #     self.net.to(device)
    #     # 3. loop over epoch
    #     for epoch in range(num_epochs):
    #         for phase in ['train', 'val']:
    #             if phase == 'train':
    #                 self.net.train()
    #             else:
    #                 self.net.eval()
    #
    #             # 5. initialize loss per phase
    #             epoch_loss = .0
    #             epoch_correct = 0
    #
    #             # 7. iterate dataloader
    #             for input_character_indices, spectrogram_labels in tqdm(
    #                     dataloaders_dict[phase]):  # dataloader는 자체로 iterable
    #                 # 8. dataset to device
    #                 input_character_indices = input_character_indices.to(device)
    #                 spectrogram_labels = spectrogram_labels.to(device)
    #
    #                 # 9. initialize grad
    #                 optimizer.zero_grad()
    #
    #                 # 10. forward
    #                 with torch.set_grad_enabled(
    #                         mode=(phase == 'train')):  # enable grad only when training # with + context_manager
    #                     # Encoder
    #                     encoder_output, (encoder_h_n, encoder_c_n) = self.encoder.forward(input_character_indices)
    #                     # Attention&Decoder
    #                     self.attention.h = encoder_output  # attention.h.Size([input length, batch, encoder output units])
    #                     self.decoder.reset(batch_size)
    #                     h_prev_1, stop_token_cum = self.decoder.h_prev_1, self.decoder.stop_token_cum  # Local variable to speed up
    #                     for decoder_step in range(self.decoder.max_output_time_length):
    #                         print('\n---------------------', 'decoder step: ', decoder_step + 1)
    #                         context_vector = self.attention.forward(h_prev_1, stop_token_cum)
    #                         h_prev_1, stop_token_cum = self.decoder.forward(context_vector)
    #                         if not any(stop_token_cum):  # stop decoding if no further prediction is needed for any samples in batch
    #                             break
    #
    #                     # Calc loss
    #                     loss = criterion(self.decoder.spectrogram_pred, spectrogram_labels)
    #
    #                     # 11. (training)calc grad
    #                     if phase == 'train':
    #                         loss.backward()
    #                         # 12. (training)update parameters
    #                         optimizer.step()
    #
    #                     # 13. add loss and correct per minibatch per phase
    #                     epoch_loss += loss.item() * input_character_indices.size(0)
    #
    #         # 14. print epoch summary
    #         epoch_loss /= len(dataloaders_dict[phase].dataset)  ## len(dataloader): num of datum
    #
    #         print('Epoch loss: {:.4f}'.format(epoch_loss))


# def checkup():
#     taco = Tacotron2()
#     criterion = tacotron.loss_function.Taco2Loss()
#     optimizer = torch.optim.Adam(taco.parameters())
#     taco.pseudo_train(criterion=criterion,
#                       optimizer=optimizer,
#                       num_epochs=3)
#
#
# checkup()
