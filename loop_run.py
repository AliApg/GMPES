import ctypes

from gmpes import (  # type:ignore pylint:disable=E0611
    GMPES,
    listdir_full,
    print_results_box,
)

if __name__ == "__main__":
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

    for t5_folder in listdir_full(
        r"D:\Uni\Thesis\Codes\Results\Embeddings\Size 768\T5 embeddings"
    ):
        gmpes = GMPES(
            model_files_path=[
                r"D:\Uni\Thesis\Codes\Results\Embeddings\Size 768\SimCSE embeddings",
                t5_folder,
            ],
            save_folder_path=r"D:\Uni\Thesis\Codes\Results\Combined",
            population_size=50,
            num_generations=50,
        )

        for ds_name in [
            "stsb",
            "sts12",
            "sts13",
            "sts14",
            "sts15",
            "sts16",
            "sickr",
        ]:

            metrics = gmpes.run(ds_name)
            print_results_box(metrics, 56, True)

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
