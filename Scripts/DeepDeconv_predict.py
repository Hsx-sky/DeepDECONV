import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat


from transformer_ModelParaSet import maxLen, inputCharacter
from DataLoad.embedding2tensor import dataAddPadReg
from ModelConstruct.transformer2head_model0 import Transformer
from ModelConstruct.transformer2head_regClassify_train_predict0 import predict


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run DeepDECONV prediction on RNA fluorescence traces."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input .npy file containing a dictionary of RNA traces."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model weights."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--max-signal-len",
        type=int,
        default=256
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max-polii-contribution",
        type=int,
        default=30
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save prediction plots."
    )


    return parser.parse_args()



# ============================================================
# Device
# ============================================================

def get_device(device_arg):

    if device_arg == "auto":

        return torch.device(
            "cuda:0" if torch.cuda.is_available()
            else "cpu"
        )


    if device_arg == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA was requested but is not available."
        )


    return torch.device(device_arg)



# ============================================================
# Normalize RNA trace
# ============================================================

def normalize_trace(trace):

    trace = np.asarray(
        trace,
        dtype=float
    )


    if np.max(trace) <= 0:

        return np.zeros_like(
            trace,
            dtype=int
        )


    trace = (
        trace /
        np.max(trace)
        *
        (inputCharacter - 1)
    )


    trace = np.clip(
        trace,
        0,
        1000
    )


    return trace.astype(int)



# ============================================================
# Prediction
# ============================================================

def run_prediction(
        signal_series,
        model,
        max_signal_len,
        overlap,
        max_polii_contribution
):


    signal_len, n_traces = signal_series.shape


    polii_contri = np.zeros(
        (
            max_polii_contribution,
            n_traces
        )
    )


    polii_ini = np.zeros(
        (
            signal_len,
            n_traces
        )
    )


    rna_input = np.zeros(
        (
            signal_len,
            n_traces
        )
    )


    rna_reconstructed = np.zeros(
        (
            signal_len,
            n_traces
        )
    )



    for i in range(n_traces):


        signal = normalize_trace(
            signal_series[:, i]
        )


        start_i = 0
        end_i = 0
        index = 0



        while end_i != signal_len:


            index += 1


            end_i = int(
                min(
                    start_i + max_signal_len,
                    signal_len
                )
            )


            start_i_next = int(
                max(
                    end_i - overlap,
                    start_i + 1
                )
            )


            cover_add = int(
                overlap / 2
            )


            if index == 1:

                cover_add = 0



            cover_start = start_i + cover_add



            signal_chunk = signal[
                start_i:end_i
            ]



            input_x = dataAddPadReg(
                signal_chunk,
                maxLen
            )



            with torch.no_grad():

                yp0_tensor, yp1_tensor = predict(
                    input_x,
                    model
                )



            # -------------------------------
            # input RNA signal
            # -------------------------------

            x00 = np.array(
                input_x[
                    input_x != -float("inf")
                ]
                .tolist()
            )


            x00 = np.array(
                [
                    int(x)
                    for x in x00
                ]
            )[1:-1]



            # -------------------------------
            # contribution
            # -------------------------------

            yp0 = (
                yp0_tensor
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


            yp0 = np.array(
                (
                    yp0 +
                    [0] *
                    max_polii_contribution
                )
                [
                    1:
                    max_polii_contribution + 1
                ]
            )


            yp0[yp0 < 0] = 0



            polii_contri[:, i] = yp0



            # -------------------------------
            # initiation
            # -------------------------------

            yp1 = (
                yp1_tensor
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


            yp1 = np.array(
                (
                    yp1 +
                    [0] *
                    max_signal_len
                )
                [
                    1:
                    max_signal_len + 2
                ]
            )



            yp1_add = yp1[
                1 + cover_add:
                len(x00) + 1
            ]



            polii_ini[
                cover_start:end_i,
                i
            ] = yp1_add[
                :
                end_i - cover_start
            ]



            # -------------------------------
            # reconstruct RNA
            # -------------------------------

            x_rec = np.convolve(
                yp1[1:],
                yp0,
                mode="full"
            )


            x_rec = x_rec[
                :len(x00)
            ]



            if np.max(x_rec) >= 1:


                x_rec = np.round(
                    x_rec *
                    np.mean(x00)
                    /
                    np.mean(x_rec)
                ).astype(int)


            else:

                x_rec = np.round(
                    x_rec
                ).astype(int)



            if np.mean(
                np.array(input_x)[1:-1]
            ) <= 0:


                x_rec[
                    x_rec != 0
                ] = 0



            rna_reconstructed[
                cover_start:end_i,
                i
            ] = x_rec[
                cover_add:
                end_i-cover_start+cover_add
            ]



            rna_input[
                cover_start:end_i,
                i
            ] = x00[
                cover_add:
                end_i-cover_start+cover_add
            ]



            start_i = start_i_next



    return (
        polii_ini,
        polii_contri,
        rna_input,
        rna_reconstructed
    )



# ============================================================
# Plot
# ============================================================

def plot_prediction_result(
        rna_input,
        rna_reconstructed,
        polii_ini,
        polii_contri,
        name,
        output_dir,
        time_res=1
):


    if rna_input.ndim == 2:

        rna_input = rna_input[:,0]


    if rna_reconstructed.ndim == 2:

        rna_reconstructed = rna_reconstructed[:,0]


    if polii_ini.ndim == 2:

        polii_ini = polii_ini[:,0]



    time_step = (
        np.arange(
            len(rna_input)
        )
        *
        time_res
    )



    plt.figure(
        figsize=(8,8)
    )


    # --------------------------
    # RNA signal
    # --------------------------

    plt.subplot(
        3,
        1,
        1
    )


    plt.plot(
        time_step,
        rna_input,
        linewidth=2,
        label="Original"
    )


    plt.plot(
        time_step,
        rna_reconstructed,
        linestyle="-.",
        linewidth=2,
        label="Reconstructed"
    )


    plt.ylabel(
        "RNA signal"
    )

    plt.title(
        name
    )

    plt.legend(
        frameon=False
    )



    # --------------------------
    # initiation
    # --------------------------

    plt.subplot(
        3,
        1,
        2
    )


    plt.bar(
        time_step,
        polii_ini,
        width=time_res
    )


    plt.ylabel(
        "PolII initiation"
    )



    # --------------------------
    # contribution
    # --------------------------

    plt.subplot(
        3,
        1,
        3
    )


    contr_time = (
        np.arange(
            len(polii_contri)
        )
        *
        time_res
    )


    plt.plot(
        contr_time,
        polii_contri,
        linestyle="-."
    )


    plt.ylabel(
        "PolII contribution"
    )


    plt.xlabel(
        "Time"
    )



    plt.tight_layout()



    save_path = (
        output_dir /
        f"{name}_prediction.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



# ============================================================
# Main
# ============================================================

def main():


    args = parse_args()


    output_dir = Path(
        args.output
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )



    device = get_device(
        args.device
    )


    print(
        "Using device:",
        device
    )



    model = Transformer()


    state_dict = torch.load(
        args.model,
        map_location=device
    )


    model.load_state_dict(
        state_dict
    )


    model.to(device)

    model.eval()



    rna_dict = np.load(
        args.input,
        allow_pickle=True
    ).item()



    polii_ini_dict = {}
    polii_contri_dict = {}
    rna_input_dict = {}
    rna_reconstructed_dict = {}



    for name, signal_series in rna_dict.items():


        print(
            "Processing:",
            name
        )


        signal_series = np.asarray(
            signal_series
        )



        if signal_series.ndim == 1:

            signal_series = signal_series[:,None]



        (
            polii_ini,
            polii_contri,
            rna_input,
            rna_reconstructed

        ) = run_prediction(
            signal_series,
            model,
            args.max_signal_len,
            args.overlap,
            args.max_polii_contribution
        )



        safe_name = str(name)[:30]



        polii_ini_dict[safe_name] = polii_ini
        polii_contri_dict[safe_name] = polii_contri
        rna_input_dict[safe_name] = rna_input
        rna_reconstructed_dict[safe_name] = rna_reconstructed



        if args.plot:


            plot_prediction_result(
                rna_input,
                rna_reconstructed,
                polii_ini,
                polii_contri,
                safe_name,
                output_dir
            )



    np.save(
        output_dir/"polII_ini.npy",
        polii_ini_dict
    )


    np.save(
        output_dir/"polII_contri.npy",
        polii_contri_dict
    )


    np.save(
        output_dir/"RNA_input.npy",
        rna_input_dict
    )


    np.save(
        output_dir/"RNA_reconstructed.npy",
        rna_reconstructed_dict
    )



    savemat(
        output_dir/"DeepDECONV_prediction.mat",
        {
            "polII_ini":polii_ini_dict,
            "polII_contri":polii_contri_dict,
            "RNA_input":rna_input_dict,
            "RNA_reconstructed":rna_reconstructed_dict
        }
    )


    print(
        "Prediction finished."
    )


    print(
        "Results saved to:",
        output_dir
    )



if __name__ == "__main__":

    main()
