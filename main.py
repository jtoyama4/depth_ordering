import numpy as np
from net import Encoder, ColorNet, SetmentationNet

if __name__ == "__main__":
    
    encoder = Encoder()
    color_decoder = ColorNet(args.feature_dim)
    segmentator = SegmentationNet(args.feature_dim)
    criterion = nn.MSELoss()
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr, weight_decay=1e-5)
    color_decoder_optim = torch.optim.Adam(color_decoder.parameters(), lr=args.decoder_lr, weight_decay=1e-5)
    segment_optim = torch.optim.Adam(segmentator.parameters(), lr=args.segmentator_lr, weight_decay=1e-5)

    data_loader = []

    for n, batch in enumerate(data_loader):
        feature = encoder(batch)
        
