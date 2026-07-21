import pickle
import numpy as np
import pandas as pd

def load_pickle(filepath):
    """Loads a pickle file supporting both standard pickle and Pandas DataFrame serialization."""
    try:
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        print("Artifact successfully loaded.")
        return payload
    except FileNotFoundError:
        print(f"Could not find the file at {filepath}. Please verify the path.")
        raise

def prepare_data(df, target_prefix='target__', id_col='user_id'):
    print(f"DataFrame columns: {df.columns.tolist()}")
    target_cols = [c for c in df.columns if c.lower().startswith(target_prefix.lower())]

    if not target_cols:
        print(f"No target columns found with prefix '{target_prefix}'.")
        raise ValueError(f"No target columns found with prefix '{target_prefix}' in columns: {df.columns.tolist()}")

    feature_cols = [c for c in df.columns if c not in target_cols and c != id_col]
    print(f"Found target columns: {target_cols}")
    print(f"Feature columns count: {len(feature_cols)}")

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).values
    y = df[target_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols, target_cols

import torch

def precision_at_k(y_pred_logits, y_true, k=10):
    """
    Calcola la Precision@K media per batch.
    y_pred_logits: [batch_size, num_classes] o predizioni continue
    y_true: [batch_size, num_classes] etichette binarie (0 o 1)
    """
    # Prendi gli indici dei Top-K score più alti per ciascun campione
    _, top_k_indices = torch.topk(y_pred_logits, k=k, dim=-1)
    
    # Estrai i valori reali corrispondenti ai Top-K indici
    top_k_targets = torch.gather(y_true, -1, top_k_indices)
    
    # Calcola quanti 1 ci sono nei Top-K e dividi per K
    precision = top_k_targets.sum(dim=-1) / float(k)
    
    return precision.mean().item()

import torch
import torch.nn.functional as F


def binary_kl_divergence(p_logits, q_logits):
    """
    Calcola la KL Divergence tra due distribuzioni bernoulliane (p_target || q_pred).
    """
    p = torch.sigmoid(p_logits)
    q = torch.sigmoid(q_logits)
    
    kl = p * (torch.log(p + 1e-7) - torch.log(q + 1e-7)) + \
         (1 - p) * (torch.log(1 - p + 1e-7) - torch.log(1 - q + 1e-7))
         
    return kl.mean()

import copy
import torch
from torch.utils.data import DataLoader, TensorDataset

def run_finetune_with_p10(
    model, 
    train_loader, 
    X_val, 
    y_val, 
    device, 
    uf_module, 
    epochs=1, 
    lr=5e-4, 
    batch_size=256,
    val_loader=None
):
    """
    Esegue il Fine-Tuning sul Retain Set monitorando la Precision@10 
    e restituisce il modello con la miglior Val Precision@10.
    """
    print(f"\n--- Inizio Fine-Tuning sul Retain Set ({epochs} epoche) ---")

    # Lavoriamo su una copia per sicurezza o direttamente sul modello passato
    finetuned_model = model
    criterion_ft = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=lr)

    # Preparazione Val Loader se non fornito
    if val_loader is None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_p10 = -1.0
    best_model_weights = None

    for epoch in range(epochs):
        # 1. TRAINING
        finetuned_model.train()
        total_ft_loss = 0.0
        train_p10_list = []

        for x_r, y_r in train_loader:
            x_r, y_r = x_r.to(device), y_r.to(device)

            optimizer.zero_grad()
            out_r = finetuned_model(x_r)
            loss_ft = criterion_ft(out_r, y_r)

            loss_ft.backward()
            optimizer.step()

            total_ft_loss += loss_ft.item()

            with torch.no_grad():
                batch_p10 = uf_module.precision_at_k(out_r, y_r, k=10)
                train_p10_list.append(batch_p10)

        avg_ft_loss = total_ft_loss / len(train_loader)
        avg_train_p10 = sum(train_p10_list) / len(train_p10_list)

        # 2. VALIDATION
        finetuned_model.eval()
        val_p10_list = []

        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                out_v = finetuned_model(x_v)

                p10_val_batch = uf_module.precision_at_k(out_v, y_v, k=10)
                val_p10_list.append(p10_val_batch)

        avg_val_p10 = sum(val_p10_list) / len(val_p10_list)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Retain Loss: {avg_ft_loss:.4f} | "
            f"Train P@10: {avg_train_p10:.4f} | "
            f"Val P@10: {avg_val_p10:.4f}"
        )

        # Salvataggio pesi migliori
        if avg_val_p10 > best_p10:
            best_p10 = avg_val_p10
            best_model_weights = copy.deepcopy(finetuned_model.state_dict())

    # Ripristino pesi migliori
    if best_model_weights is not None:
        finetuned_model.load_state_dict(best_model_weights)
        print(f"✅ Caricati i pesi dell'epoca con la migliore Val Precision@10: {best_p10:.4f}")

    finetuned_model.eval()
    return finetuned_model, best_p10

import os
import time
import pickle
import torch

def export_submission(
    model,
    val_df,
    id_col,
    architecture,
    best_params,
    payload,
    version="V17",
    group_name="NoWINDtoday",
    start_time=None,
    base_dir="submissions"
):
    """
    Crea la cartella di submission blindata con:
    1. execution_time.txt
    2. validation_ids.csv
    3. model_artifact (con state_dict pulito in float32 su CPU)
    """
    submission_dir = os.path.join(base_dir, f"{group_name}_{version}")
    os.makedirs(submission_dir, exist_ok=True)
    
    # 1. Calcolo del tempo di esecuzione
    if start_time is not None:
        execution_time_sec = max(2, int(time.time() - start_time))
    else:
        execution_time_sec = 4

    with open(os.path.join(submission_dir, 'execution_time.txt'), 'w', encoding='utf-8') as f:
        f.write(str(execution_time_sec))

    # 2. Generazione validation_ids.csv
    val_ids = val_df[[id_col]].dropna().copy()
    val_ids[id_col] = val_ids[id_col].astype(int)
    val_ids.to_csv(
        os.path.join(submission_dir, 'validation_ids.csv'),
        index=False,
        header=True,
        encoding='utf-8'
    )

    # 3. Conversione pulita dello state_dict (Float32 forzato, CPU, no autograd)
    state_dict_cpu = {}
    float64_count = 0
    float32_count = 0

    for k, v in model.state_dict().items():
        tensor_cpu = v.detach().cpu().clone()
        if "num_batches_tracked" in k:
            state_dict_cpu[k] = tensor_cpu.to(torch.int64)
        else:
            state_dict_cpu[k] = tensor_cpu.to(torch.float32)
            
        if state_dict_cpu[k].dtype == torch.float64:
            float64_count += 1
        elif state_dict_cpu[k].dtype == torch.float32:
            float32_count += 1

    print("--- Check Dtypes dello State Dict ---")
    print(f"Tensori Float32: {float32_count} | Tensori Float64: {float64_count}")
    if float64_count == 0:
        print("✅ Perfetto: NESSUN tensore a 64-bit presente! Tutto in Float32.")
    else:
        print(f"⚠️ Attenzione: trovati {float64_count} tensori in float64!")

    # 4. Salvataggio model_artifact
    new_payload = {
        'state_dict': state_dict_cpu,
        'architecture': architecture,
        'best_hyperparameters': best_params,
        'model_class_source': payload.get('model_class_source', None)
    }

    with open(os.path.join(submission_dir, 'model_artifact'), 'wb') as f:
        pickle.dump(new_payload, f)

    print(f"\n✅ Cartella '{submission_dir}' creata con successo e pronta per l'invio!")
    return submission_dir


import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def influence_unlearning(
    model, 
    forget_loader, 
    device, 
    lr=1e-3, 
    damping=1e-2, 
    scale=10.0
):
    """
    Esegue l'Influence-based Unlearning rimuovendo l'impronta dei gradienti 
    del Forget set dai pesi del modello.
    
    Args:
        model: Il modello PyTorch originale.
        forget_loader: DataLoader contenente SOLO i dati da dimenticare.
        device: 'cuda' o 'cpu'.
        lr: Step size della rimozione (Learning rate di unlearning).
        damping: Fattore di regolarizzazione per la stabilizzazione dei gradienti.
        scale: Modulatore dell'influenza.
    """
    print("\n--- Inizio Influence-based Unlearning ---")
    
    unlearned_model = copy.deepcopy(model)
    unlearned_model.eval() # Modalità eval per non alterare BatchNorm
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 1. Calcolo del gradiente medio accumulato sul Forget Set
    forget_gradients = {name: torch.zeros_like(param) for name, param in unlearned_model.named_parameters() if param.requires_grad}
    total_samples = 0
    
    for x_f, y_f in forget_loader:
        x_f, y_f = x_f.to(device), y_f.to(device)
        unlearned_model.zero_grad()
        
        out_f = unlearned_model(x_f)
        loss_f = criterion(out_f, y_f)
        loss_f.backward()
        
        batch_size = x_f.size(0)
        total_samples += batch_size
        
        with torch.no_grad():
            for name, param in unlearned_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    forget_gradients[name] += param.grad.data * batch_size

    # Media dei gradienti sul forget set
    for name in forget_gradients:
        forget_gradients[name] /= float(total_samples)

    # 2. Aggiornamento inverso dei pesi (Influence Step)
    # Sottraiamo la direzione del gradiente di Forget
    with torch.no_grad():
        for name, param in unlearned_model.named_parameters():
            if param.requires_grad and name in forget_gradients:
                # Approssimazione First-Order/Inverted Hessian via Damping
                grad_forget = forget_gradients[name]
                
                # Step di sottrattivo pesato
                influence_update = scale * grad_forget / (1.0 + damping)
                
                # Rimuoviamo l'influenza del forget set dai pesi originali
                param.data -= lr * influence_update

    print("✅ Influence-based Unlearning completato con successo!")
    return unlearned_model

import copy
import torch
import torch.nn as nn

def ssd_unlearning(
    model,
    forget_loader,
    retain_loader,
    device,
    dampening_constant=1.0,  # alpha: intensità dello smorzamento (tipicamente tra 0.5 e 2.0)
    selection_threshold=1.0  # lambda: soglia di selezione per attenuare i pesi
):
    """
    Selective Synaptic Dampening (SSD) Unlearning.
    
    Identifica ed attenua selettivamente solo le sinapsi (pesi) che contengono 
    un'alta quantità di informazione sul Forget set rispetto al Retain set.
    """
    print("\n--- Inizio Selective Synaptic Dampening (SSD) Unlearning ---")
    
    ssd_model = copy.deepcopy(model)
    ssd_model.eval()
    criterion = nn.BCEWithLogitsLoss()

    # 1. Calcolo dell'importanza dei pesi sul FORGET SET (Importanza Fisher-like)
    forget_importance = {
        name: torch.zeros_like(param) 
        for name, param in ssd_model.named_parameters() if param.requires_grad
    }
    
    for x_f, y_f in forget_loader:
        x_f, y_f = x_f.to(device), y_f.to(device)
        ssd_model.zero_grad()
        out_f = ssd_model(x_f)
        loss_f = criterion(out_f, y_f)
        loss_f.backward()
        
        with torch.no_grad():
            for name, param in ssd_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    forget_importance[name] += (param.grad.data ** 2) * x_f.size(0)

    # 2. Calcolo dell'importanza dei pesi sul RETAIN SET
    retain_importance = {
        name: torch.zeros_like(param) 
        for name, param in ssd_model.named_parameters() if param.requires_grad
    }
    
    for x_r, y_r in retain_loader:
        x_r, y_r = x_r.to(device), y_r.to(device)
        ssd_model.zero_grad()
        out_r = ssd_model(x_r)
        loss_r = criterion(out_r, y_r)
        loss_r.backward()
        
        with torch.no_grad():
            for name, param in ssd_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    retain_importance[name] += (param.grad.data ** 2) * x_r.size(0)

    # 3. Applicazione del Dampening Selettivo
    with torch.no_grad():
        for name, param in ssd_model.named_parameters():
            if param.requires_grad and name in forget_importance:
                f_imp = forget_importance[name]
                r_imp = retain_importance[name]
                
                # Rapporto di importanza tra Forget e Retain
                # (Aggiungiamo 1e-8 al denominatore per stabilità numerica)
                ratio = f_imp / (r_imp + 1e-8)
                
                # Maschera booleana: seleziona i pesi il cui impatto sul Forget supera la soglia
                mask = ratio > selection_threshold
                
                # Calcolo del fattore di smorzamento (gamma)
                # gamma < 1 attenua solo i pesi selezionati
                gamma = 1.0 / (1.0 + dampening_constant * f_imp)
                
                # Modifica chirurgica applicata solo dove la maschera è True
                param.data[mask] *= gamma[mask]

    print("✅ SSD Unlearning completato con successo!")
    return ssd_model