from sample.tester.test_enc_dec import Encoder_Tester


from sample.models.encoder_decoder import Encoder_Decoder


model=Encoder_Decoder()
output="data/checkpoints"
name="enc_dec_S15678_no_rot"



pose =Encoder_Tester( model, output, name)

#pose.test_on_test()
pose.test_on_train()