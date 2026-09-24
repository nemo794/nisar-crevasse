# Large Model Files

Two U-Net model weight files exceed GitHub's 100MB file size limit and are not included in this repository:

- `models/nisar/unet/unet_025_019_f421_meansoft_g3/unet_best.safetensors` (153MB)
- `models/biomass/unet/bio_unet_4ch_frangi_aux_nw0p02/unet_best.safetensors` (199MB)

## Options to Obtain These Files

### Option 1: Download Pre-trained Weights
These files can be provided separately upon request. Contact the repository maintainer.

### Option 2: Retrain from Scratch
Follow the instructions in the documentation to retrain the models:
- NISAR: See `docs/nisar/UNET_PIPELINE.md`
- BIOMASS: See `docs/biomass/UNET_PIPELINE.md`

## File Locations (when obtained)

Place the downloaded or retrained files in these exact locations:
```
models/nisar/unet/unet_025_019_f421_meansoft_g3/unet_best.safetensors
models/biomass/unet/bio_unet_4ch_frangi_aux_nw0p02/unet_best.safetensors
```

The other model files in this repository (gate classifiers and metadata JSON files) are all under 100MB and are included.
