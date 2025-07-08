document.addEventListener('DOMContentLoaded', () => {
    // Riferimenti agli elementi DOM
    const networkFileInput = document.getElementById('networkFile');
    const fileNameSpan = document.getElementById('fileName');
    const modelNameInput = document.getElementById('modelName');
    const uploadButton = document.getElementById('uploadButton');
    const uploadStatus = document.getElementById('uploadStatus');
    // Rimozione della progress bar simulata, poiché non è gestita dal server
    // const uploadProgressBar = document.getElementById('uploadProgressBar');
    const uploadedModelsList = document.getElementById('uploadedModelsList');
    const datasetSelect = document.getElementById('datasetSelect');
    const lowerMetricsInput = document.getElementById('lowerMetrics');
    const higherMetricsInput = document.getElementById('higherMetrics');
    
    let selectedFile = null; // Variabile per il file selezionato

    // --- Funzioni di Utilità ---

    // Aggiorna il messaggio di stato visibile all'utente
    function updateStatus(message, type = 'info') {
        uploadStatus.textContent = message;
        uploadStatus.className = `status-message ${type}`; // Aggiunge classi per stile (es. success, error)
    }

    // --- Gestione Caricamento File ---

    // Aggiorna il nome del file selezionato e abilita/disabilita il pulsante di upload
    networkFileInput.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            fileNameSpan.textContent = selectedFile.name;
            uploadButton.disabled = false;
        } else {
            fileNameSpan.textContent = 'Nessun file selezionato';
            uploadButton.disabled = true;
        }
    });

    // Funzione per caricare la rete neurale al click del pulsante
    uploadButton.addEventListener('click', async () => {
        if (!selectedFile) {
            updateStatus('Per favore, seleziona un file di rete neurale.', 'error');
            return;
        }

        updateStatus('Caricamento in corso...', 'info');
        uploadButton.disabled = true;
        // uploadProgressBar.style.display = 'block'; // Non più necessaria
        // uploadProgressBar.value = 0; // Non più necessaria

        const formData = new FormData();
        formData.append('network_file', selectedFile);
        formData.append('model_name', modelNameInput.value || 'Modello Senza Nome');
        formData.append('dataset', datasetSelect.value);

        // Parsing delle metriche (lower e higher)
        function parseMetrics(textareaValue) {
            const lines = textareaValue.trim().split('\n');
            const metrics = {};
            lines.forEach(line => {
                const [key, value] = line.split(':').map(s => s.trim());
                if (key && value && !isNaN(value)) {
                    metrics[key] = parseFloat(value);
                }
            });
            return metrics;
        }
        
        const lowerMetrics = parseMetrics(lowerMetricsInput.value);
        const higherMetrics = parseMetrics(higherMetricsInput.value);
        
        formData.append('lower_metrics', JSON.stringify(lowerMetrics));
        formData.append('higher_metrics', JSON.stringify(higherMetrics));
        
        try {
            const response = await fetch('http://localhost:5000/upload-neural-network', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json(); // Il server dovrebbe sempre rispondere JSON

            if (response.ok) {
                updateStatus('Caricamento completato!', 'success');
                // Ricarica la lista dei modelli per mostrare lo stato aggiornato
                await loadUploadedModels();
            } else {
                // Se la risposta non è OK, 'result' conterrà l'oggetto errore dal server
                updateStatus(`Errore nel caricamento: ${result.error || 'Errore sconosciuto'}`, 'error');
            }
        } catch (error) {
            updateStatus(`Errore di rete: ${error.message}`, 'error');
            console.error('Errore durante il caricamento:', error);
        } finally {
            // uploadProgressBar.style.display = 'none'; // Non più necessaria
            uploadButton.disabled = false;
            // Resetta i campi di input
            networkFileInput.value = '';
            fileNameSpan.textContent = 'Nessun file selezionato';
            modelNameInput.value = '';
            selectedFile = null;
        }
    });

    // --- Gestione Lista Modelli Caricati ---
//     <div class="model-details" id="details-${model.id}" style="display:none;">
//     <h4>Dettagli Ispezione:</h4>
//     <pre>${JSON.stringify(model.details, null, 2)}</pre>
// </div>
    // Funzione per caricare e visualizzare la lista dei modelli dal back-end
    async function loadUploadedModels() {
        try {
            const response = await fetch('http://localhost:5000/neural-networks');
            const models = await response.json();

            uploadedModelsList.innerHTML = ''; // Pulisci la lista esistente

            if (models.length === 0) {
                uploadedModelsList.innerHTML = '<p>Nessuna rete neurale caricata al momento.</p>';
                return;
            }

            models.forEach(model => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <div class="model-header">
                        <div class="model-info">
                            <strong>${model.name}</strong> (ID: ${model.id})
                            <br>Status: <span class="status-${model.status.toLowerCase().replace('_', '-')}">${model.status.replace('_', ' ')}</span>
                        </div>
                        <div class="model-actions">
                            ${model.status === 'ready' ? `<button class="run-button" data-id="${model.id}">Esegui</button>` : ''}
                            ${model.status === 'running' ? `<button class="check-result-button" data-id="${model.id}">Controlla Risultato</button>` : ''}
                        </div>
                    </div>
                    ${model.error ? `<div class="error-message">Errore: ${model.error}</div>` : ''}

                `;
                uploadedModelsList.appendChild(li);
               if (model.status === 'completed') {

                li.querySelector('.model-header').addEventListener('click', async () => {
                const container = document.getElementById(`result-${model.id}`);
                if (container.innerHTML !== '') {
                    container.style.display = container.style.display === 'none' ? 'block' : 'none';
                    return;
                }

                try {
                    const response = await fetch(`http://localhost:5000/execution-result/${model.id}`);
                    const result = await response.json();

                    if (response.ok && result.status === 'completed') {
                    const metrics = result.results.metrics;
                    const downloadUrl = result.results.optimized_model_path;

                    const percentReduction = (orig, opt) => ((1 - opt / orig) * 100).toFixed(2);

                    const formatted = `
                        <div class="execution-result">
                        <h4>📊 Risultati Ottimizzazione</h4>
                        <table>
                            <tr><th>Parametro</th><th>Originale</th><th>Ottimizzato</th><th>Riduzione</th></tr>
                            <tr><td><strong>Inference Time (s)</strong></td><td>${metrics.original_inference_time.toFixed(4)}</td><td>${metrics.inference_time.toFixed(4)}</td><td>${percentReduction(metrics.original_inference_time, metrics.inference_time)}%</td></tr>
                            <tr><td><strong>Dimensione (MB)</strong></td><td>${metrics.original_size.toFixed(2)}</td><td>${metrics.optimized_size.toFixed(2)}</td><td>${percentReduction(metrics.original_size, metrics.optimized_size)}%</td></tr>
                            <tr><td><strong>Reduction Ratio</strong></td><td>—</td><td>${metrics.reduction_ratio.toFixed(2)}</td><td>—</td></tr>
                        </table>
                        <a href="${downloadUrl}" download class="download-link">⬇️ Scarica modello ottimizzato</a>
                        </div>
                    `;

                    container.innerHTML = formatted;
                    container.style.display = 'block';
                    }
                } catch (err) {
                    updateStatus(`Errore nel recupero dei risultati per ${model.id}: ${err.message}`, 'error');
                }
                });
            }
                // Aggiungi un listener per mostrare/nascondere i dettagli al click sull'header
                li.querySelector('.model-header').addEventListener('click', () => {
                    const detailsDiv = document.getElementById(`details-${model.id}`);
                    detailsDiv.style.display = detailsDiv.style.display === 'none' ? 'block' : 'none';
                });
            });

            // Aggiungi listener ai pulsanti "Esegui" appena creati
            document.querySelectorAll('.run-button').forEach(button => {
                button.addEventListener('click', (event) => {
                    event.stopPropagation(); // Evita che il click si propaghi all'header
                    const modelId = event.target.dataset.id;
                    runNeuralNetwork(modelId);
                });
            });

            // Aggiungi listener ai pulsanti "Controlla Risultato" appena creati
            document.querySelectorAll('.check-result-button').forEach(button => {
                button.addEventListener('click', (event) => {
                    event.stopPropagation(); // Evita che il click si propaghi all'header
                    const modelId = event.target.dataset.id;
                    checkExecutionResult(modelId);
                });
            });

        } catch (error) {
            updateStatus(`Errore nel caricamento delle reti: ${error.message}`, 'error');
            console.error('Errore durante il caricamento delle reti:', error);
        }
    }

    // --- Funzioni di Esecuzione e Polling ---

    // Funzione per avviare l'esecuzione della rete neurale
    // async function runNeuralNetwork(modelId) {
    //     // Chiede all'utente i dati di input come array JSON
    //     const inputDataStr = prompt("Inserisci i dati di input come array JSON (es. [0.1, 0.2, ..., 1.0]):");
    //     if (!inputDataStr) return; // L'utente ha annullato

    //     let inputData;
    //     try {
    //         inputData = JSON.parse(inputDataStr);
    //         if (!Array.isArray(inputData)) {
    //             throw new Error("L'input deve essere un array JSON.");
    //         }
    //     } catch (e) {
    //         alert("Formato input non valido. Inserisci un array JSON corretto.");
    //         updateStatus("Input non valido.", "error");
    //         return;
    //     }

    //     updateStatus(`Avvio esecuzione per ${modelId}...`, 'info');
    //     try {
    //         const response = await fetch(`http://localhost:5000/run-neural-network/${modelId}`, {
    //             method: 'POST',
    //             headers: {
    //                 'Content-Type': 'application/json',
    //             },
    //             body: JSON.stringify({ input_data: inputData }),
    //         });

    //         const result = await response.json();

    //         if (response.ok) {
    //             updateStatus(`Esecuzione avviata per ${modelId}. Stato: ${result.status}`, 'success');
    //             await loadUploadedModels(); // Aggiorna la lista per riflettere lo stato 'running'
    //             // Potresti avviare un polling automatico qui, es. setTimeout(() => checkExecutionResult(modelId), 2000);
    //         } else {
    //             updateStatus(`Errore nell'avvio dell'esecuzione: ${result.error || 'Errore sconosciuto'}`, 'error');
    //         }
    //     } catch (error) {
    //         updateStatus(`Errore di rete durante l'avvio dell'esecuzione: ${error.message}`, 'error');
    //         console.error('Errore durante l\'avvio dell\'esecuzione:', error);
    //     }
    // }
    async function runNeuralNetwork(modelId) {
        updateStatus(`Avvio esecuzione per ${modelId}...`, 'info');

        // Trova il bottone premuto
        const runButton = document.querySelector(`.run-button[data-id="${modelId}"]`);
        if (runButton) {
            runButton.disabled = true;
            runButton.textContent = 'Processing...';
        }

        let metricsLower, metricsHigher, dataset;
        try {
            function parseMetrics(textareaValue) {
                const lines = textareaValue.trim().split('\n');
                const metrics = {};
                lines.forEach(line => {
                    const [key, value] = line.split(':').map(s => s.trim());
                    if (key && value && !isNaN(value)) {
                        metrics[key] = parseFloat(value);
                    }
                });
                return metrics;
            }

            metricsLower = parseMetrics(document.getElementById('lowerMetrics').value || '');
            metricsHigher = parseMetrics(document.getElementById('higherMetrics').value || '');
            dataset = document.getElementById('datasetSelect').value;

            if (!dataset) throw new Error("Dataset non selezionato.");

        } catch (e) {
            alert("Errore nei dati inseriti: " + e.message);
            updateStatus("Metriche o dataset non validi.", "error");
            if (runButton) {
                runButton.disabled = false;
                runButton.textContent = 'Esegui';
            }
            return;
        }

        try {
            const response = await fetch(`http://localhost:5000/run-neural-network/${modelId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    metrics: { lower: metricsLower, higher: metricsHigher },
                    dataset: dataset
                }),
            });

            const result = await response.json();

            if (response.ok) {
                updateStatus(`Esecuzione avviata per ${modelId}. Stato: ${result.status}`, 'success');
            } else {
                updateStatus(`Errore nell'avvio dell'esecuzione: ${result.error || 'Errore sconosciuto'}`, 'error');
            }
        } catch (error) {
            updateStatus(`Errore di rete durante l'avvio dell'esecuzione: ${error.message}`, 'error');
            console.error('Errore durante l\'avvio dell\'esecuzione:', error);
        } finally {
            // Ricarica lo stato aggiornato della lista, che mostrerà "Controlla Risultato"
            await loadUploadedModels();
        }
    }

    
    // Funzione per controllare i risultati dell'esecuzione (polling manuale)
    async function checkExecutionResult(modelId) {
        updateStatus(`Controllo risultati per ${modelId}...`, 'info');
        try {
            const response = await fetch(`http://localhost:5000/execution-result/${modelId}`);
            const result = await response.json();

            if (response.ok) {
                if (result.status === 'completed') {
                    const metrics = result.results.metrics;
                    const downloadUrl = result.results.optimized_model_path;

                    const percentReduction = (orig, opt) => {
                        return ((1 - opt / orig) * 100).toFixed(2);
                    };

                    const formattedResults = `
                        <div class="execution-result">
                            <h4>Risultati Ottimizzazione</h4>
                            <table>
                                <tr>
                                    <th>Parametro</th>
                                    <th>Originale</th>
                                    <th>Ottimizzato</th>
                                    <th>Riduzione</th>
                                </tr>
                                <tr>
                                    <td><strong>Inference Time (s)</strong></td>
                                    <td>${metrics.original_inference_time.toFixed(4)}</td>
                                    <td>${metrics.inference_time.toFixed(4)}</td>
                                    <td>${percentReduction(metrics.original_inference_time, metrics.inference_time)}%</td>
                                </tr>
                                <tr>
                                    <td><strong>Dimensione (MB)</strong></td>
                                    <td>${metrics.original_size.toFixed(2)}</td>
                                    <td>${metrics.optimized_size.toFixed(2)}</td>
                                    <td>${percentReduction(metrics.original_size, metrics.optimized_size)}%</td>
                                </tr>
                                <tr>
                                    <td><strong>Reduction Ratio</strong></td>
                                    <td>—</td>
                                    <td>${metrics.reduction_ratio.toFixed(2)}</td>
                                    <td>—</td>
                                </tr>
                            </table>
                            <a href="${downloadUrl}" download class="download-link"> Scarica modello ottimizzato</a>
                        </div>
                    `;

                    updateStatus(`Esecuzione completata per ${modelId}.`, 'success');
                    uploadedModelsList.innerHTML = formattedResults;
                } else if (result.status === 'execution_failed') {
                    updateStatus(`Esecuzione fallita per ${modelId}: ${result.results ? JSON.stringify(result.results, null, 2) : result.error}`, 'error');
                    await loadUploadedModels(); // Aggiorna lo stato a 'execution_failed'
                } else {
                    updateStatus(`Esecuzione ancora in corso per ${modelId}. Riprova tra poco.`, 'info');
                    // Qui potresti anche ri-programmare un altro controllo, es. setTimeout(() => checkExecutionResult(modelId), 2000);
                }
            } else {
                updateStatus(`Errore nel recupero risultati: ${result.error || 'Errore sconosciuto'}`, 'error');
            }
        } catch (error) {
            updateStatus(`Errore di rete durante il recupero risultati: ${error.message}`, 'error');
            console.error('Errore durante il recupero dei risultati:', error);
        }
    }

    // --- Inizializzazione ---
    // Carica la lista dei modelli quando la pagina è completamente caricata
    loadUploadedModels();
});
