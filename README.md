# Student burnout risk estimator

This project includes a static web app for Vercel. The app runs model
predictions entirely in the visitor's browser; it has no prediction API or
server-side form submission.

## Deploy to Vercel

1. Import this GitHub repository in the Vercel dashboard.
2. Leave the framework preset as **Other** and the output directory as
   `public` (also configured in `vercel.json`).
3. Deploy. The resulting Vercel URL serves the app at `/`.

The browser app loads `public/burnout_model.json.gz`. The compressed model is
small enough to keep with the source and does not require Python or
scikit-learn at runtime. The `models/*.joblib` training artifact is
intentionally ignored by Git because it is not used by the static deployment.

## Use the web app locally

From the project directory, serve the `public` folder over HTTP:

```powershell
py -m http.server 8000 --directory public
```

Then open [http://localhost:8000](http://localhost:8000). Opening `index.html`
directly as a `file://` URL will not work because browsers block fetching the
compressed model from local files.

## Retrain and export

Install the training dependencies listed in `requirements.txt`, then run:

```powershell
python train_model.py
```

The script compares candidate classifiers using a separate validation split,
evaluates the selected full-size model and a compact 128-tree browser model on
an untouched test split, and writes the gzip-compressed browser model to
`public/burnout_model.json.gz`. It saves metrics and validation results under
`models/`; the optional full-size scikit-learn pipeline is saved locally as
`models/burnout_model.joblib` and excluded from Git.

## Evaluation and limitations

The most recent evaluation is recorded in `models/metrics.json` after running
the training script. The browser edition has its own test metrics in that
file. Accuracy is modest; predictions are educational estimates, not a
clinical diagnosis or a validated high-stakes decision tool.
