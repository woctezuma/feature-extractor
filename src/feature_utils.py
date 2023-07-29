import torch
import tqdm


def extract_features(model, img_loader, batch_size, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = []
    sample_fnames = []

    with torch.no_grad():
        for ii, imgs in enumerate(tqdm.tqdm(img_loader)):
            fts = model(imgs.to(device))
            features.append(fts.cpu())
            sample_fnames += [
                img_loader.dataset.samples[ii * batch_size + jj]
                for jj in range(fts.shape[0])
            ]
            if verbose:
                print(f"Extraction: {len(features)} {len(sample_fnames)}")

    features = torch.concat(features, dim=0)

    return features, sample_fnames
