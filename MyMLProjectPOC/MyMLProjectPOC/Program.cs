using System.Text;
using Tesseract;
using UglyToad.PdfPig;
using DocumentFormat.OpenXml.Packaging;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SmartTextReader
{
    class Program
    {
        /// <summary>
        /// The main entry point for the SmartTextReader application.
        /// </summary>
        /// <remarks>This method initializes the SmartTextReader application, demonstrating text
        /// extraction from various file types such as images, PDFs, and DOCX documents. It also showcases a simple text
        /// classification using ML.NET. Ensure that the file paths specified for the sample files are correct and
        /// accessible.</remarks>
        /// <param name="args">Command-line arguments passed to the application. Currently not used.</param>
        static void Main(string[] args)
        {
            Console.WriteLine("SmartTextReader starting...");

            // Example files - change to your file paths
            var sampleImage = @"E:\Skeleton\MyMLProjectPOC\MyMLProjectPOC\SampleFiles\JPGImage.jpg";
            var samplePdf = @"E:\Skeleton\MyMLProjectPOC\MyMLProjectPOC\SampleFiles\pdfFile.pdf";
            var sampleDocx = @"E:\Skeleton\MyMLProjectPOC\MyMLProjectPOC\SampleFiles\demo.docx";

            // Extract text
            if (File.Exists(sampleImage))
                Console.WriteLine("Image text:\n" + ExtractFromImage(sampleImage));

            if (File.Exists(samplePdf))
                Console.WriteLine("PDF text:\n" + ExtractFromPdf(samplePdf));

            if (File.Exists(sampleDocx))
                Console.WriteLine("DOCX text:\n" + ExtractFromDocx(sampleDocx));

            // Example: ML.NET text classification (train on a tiny dataset)
            var mlText = "This is an invoice for payment of $2000";
            var ml = new SimpleTextClassifier();
            ml.Train(); // trains on small sample data embedded in code
            var pred = ml.Predict(mlText);
            Console.WriteLine($"ML.NET predicted label for '{mlText}': {pred}");
        }

        // -------------------------
        // Image OCR (Tesseract)
        // -------------------------
        /// <summary>
        /// Extracts text from an image file using Optical Character Recognition (OCR).
        /// </summary>
        /// <remarks>This method uses the Tesseract OCR engine to process the image. Ensure that the
        /// 'tessdata' directory is present in the application's base directory and contains the necessary language data
        /// files for OCR processing.</remarks>
        /// <param name="imagePath">The file path of the image from which to extract text. Must be a valid path to an image file.</param>
        /// <returns>The extracted text from the image. If an error occurs during processing, returns a string indicating the
        /// error.</returns>
        /// <exception cref="DirectoryNotFoundException">Thrown if the required 'tessdata' directory is not found in the application base directory.</exception>
        static string ExtractFromImage(string imagePath)
        {
            try
            {
                // Path to tessdata - relative to app base directory
                var tessDataDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "tessdata");

                // Ensure tessdata exists
                if (!Directory.Exists(tessDataDir))
                    throw new DirectoryNotFoundException($"tessdata folder not found at {tessDataDir}");

                using var engine = new TesseractEngine(tessDataDir, "eng", EngineMode.Default);
                using var img = Pix.LoadFromFile(imagePath);
                using var page = engine.Process(img);
                return page.GetText();
            }
            catch (Exception ex)
            {
                return $"[OCR error] {ex.Message}";
            }
        }

        // -------------------------
        // PDF text extraction (PdfPig)
        // -------------------------
        /// <summary>
        /// Extracts text content from a PDF file specified by the file path.
        /// </summary>
        /// <remarks>This method uses the PdfPig library to open and read the PDF document.  It iterates
        /// through each page of the document, appending the text content to a StringBuilder.</remarks>
        /// <param name="pdfPath">The file path of the PDF document from which to extract text. Must not be null or empty.</param>
        /// <returns>A string containing the extracted text from the PDF document.  If an error occurs during extraction, returns
        /// a string with an error message prefixed by "[PDF error]".</returns>
        static string ExtractFromPdf(string pdfPath)
        {
            try
            {
                var sb = new StringBuilder();
                using (var doc = PdfDocument.Open(pdfPath))
                {
                    foreach (var page in doc.GetPages())
                    {
                        sb.AppendLine(page.Text);
                    }
                }
                return sb.ToString();
            }
            catch (Exception ex)
            {
                return $"[PDF error] {ex.Message}";
            }
        }

        // -------------------------
        // DOCX text extraction (OpenXML)
        // -------------------------
        /// <summary>
        /// Extracts and returns the text content from a DOCX file specified by the given path.
        /// </summary>
        /// <remarks>This method uses the OpenXML SDK to read the DOCX file. It opens the document in
        /// read-only mode and retrieves the text from the main document part. If an error occurs during the extraction
        /// process, the method returns a string indicating the error.</remarks>
        /// <param name="docxPath">The file path of the DOCX document from which to extract text. Must not be null or empty.</param>
        /// <returns>A string containing the extracted text from the DOCX document. Returns an empty string if the document is
        /// empty or if an error occurs during extraction.</returns>
        static string ExtractFromDocx(string docxPath)
        {
            try
            {
                using var doc = WordprocessingDocument.Open(docxPath, false);
                var body = doc != null && doc.MainDocumentPart != null ? doc.MainDocumentPart.Document.Body : null;
                return body?.InnerText ?? string.Empty;
            }
            catch (Exception ex)
            {
                return $"[DOCX error] {ex.Message}";
            }
        }
    }

    // -------------------------
    // Simple ML.NET text classifier (in-memory tiny dataset)
    // -------------------------
    /// <summary>
    /// Provides functionality to train and use a simple text classification model using ML.NET.
    /// </summary>
    /// <remarks>This class allows training a text classification model with a small in-memory dataset and 
    /// predicting the category of new text inputs. The model is saved to disk after training and  can be loaded for
    /// future predictions. The default model path is set to "models/textModel.zip"  in the application's base
    /// directory.</remarks>
    public class SimpleTextClassifier
    {
        private readonly string _modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "textModel.zip");
        private readonly MLContext _mlContext = new MLContext(seed: 0);
        private ITransformer? _trainedModel;

        // Input schema
        private class ModelInput
        {
            public string Text { get; set; } = string.Empty;
            public string Label { get; set; } = string.Empty;
        }

        private class ModelOutput
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; } = string.Empty;
        }

        /// <summary>
        /// Trains a machine learning model using predefined sample data and saves the trained model to a specified
        /// path.
        /// </summary>
        /// <remarks>This method uses a small set of sample data to train a multiclass classification
        /// model.  The model is trained using the SDCA Maximum Entropy algorithm and is saved to the file system  at
        /// the path specified by <c>_modelPath</c>. Ensure that the directory exists or can be created  before calling
        /// this method.</remarks>
        public void Train()
        {
            // Tiny sample data - replace with real dataset
            var samples = new List<ModelInput>
            {
                new ModelInput { Text = "Invoice amount due for March", Label = "Invoice" },
                new ModelInput { Text = "Paid invoice for electricity", Label = "Invoice" },
                new ModelInput { Text = "Resume: Senior Software Engineer", Label = "Resume" },
                new ModelInput { Text = "Curriculum vitae and contact details", Label = "Resume" },
                new ModelInput { Text = "Monthly report for sales", Label = "Report" }
            };

            var data = _mlContext.Data.LoadFromEnumerable(samples);

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.Text)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = pipeline.Fit(data);

            // optionally save the model
            var modelDir = Path.GetDirectoryName(_modelPath);
            if (!Directory.Exists(modelDir)) Directory.CreateDirectory(modelDir ??string.Empty);
            _mlContext.Model.Save(_trainedModel, data.Schema, _modelPath);
        }

        /// <summary>
        /// Predicts the label for the given text using a trained machine learning model.
        /// </summary>
        /// <remarks>Ensure that the model is trained by calling the <c>Train</c> method before using this
        /// method. If the model is not trained, an <see cref="InvalidOperationException"/> is thrown.</remarks>
        /// <param name="text">The input text for which the prediction is to be made. Cannot be null or empty.</param>
        /// <returns>The predicted label as a string.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the model is not trained and cannot be loaded from the specified path.</exception>
        public string Predict(string text)
        {
            if (_trainedModel == null)
            {
                // try loading saved model
                if (File.Exists(_modelPath))
                    _trainedModel = _mlContext.Model.Load(_modelPath, out var schema);
                else
                    throw new InvalidOperationException("Model not trained. Call Train() first.");
            }

            var predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);
            var res = predEngine.Predict(new ModelInput { Text = text });
            return res.PredictedLabel;
        }
    }
}
