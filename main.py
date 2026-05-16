"""
Final Project: Engineering Data Systems Pipeline
Course: Computer Programming (AY 2026)
Reporting Standard: IEEE Two-Column Research Format

Student Name: Lois Yvonne Trinidad
Student Number: TUPM-25-0298
Assigned Topic ID: RMS-04 (Geothermal Heat Exchanger Fouling)
Unique Filter Logic Applied: fuel == 'Geothermal' & country == 'Philippines'
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


class GeothermalPipeline:
    """
    An Object-Oriented pipeline to ingest, clean, analyze,
    and visualize Geothermal Power Plant generation telemetry.
    """

    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.raw_data = None
        self.cleaned_data = None
        self.metrics = {}
        self.target_col = None
        self.label_col = None
        self.group_col = None

        # Ensure directory structures are active
        os.makedirs(output_dir, exist_ok=True)

    def ingest_data(self) -> pd.DataFrame:
        """Step 1: Data Ingestion optimized with Chunk-Streaming to prevent memory freezes."""
        print("[INFO] Initializing Data Ingestion Phase...")
        try:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Source file missing at {self.input_path}")
            
            print("[INFO] Streaming global file chunks into memory...")
            chunk_list = []
            
            # Read first chunk just to find the correct country/fuel headers dynamically
            first_chunk = pd.read_csv(self.input_path, nrows=5)
            cols = [c.lower() for c in first_chunk.columns]
            
            # Identify correct column casing dynamically
            fuel_header = first_chunk.columns[cols.index('primary_fuel')] if 'primary_fuel' in cols else None
            country_header = first_chunk.columns[cols.index('country_long')] if 'country_long' in cols else None
            
            if not fuel_header or not country_header:
                fuel_header = 'primary_fuel'
                country_header = 'country_long'

            for chunk in pd.read_csv(self.input_path, chunksize=5000, low_memory=False):
                if fuel_header in chunk.columns and country_header in chunk.columns:
                    filtered_chunk = chunk[
                        (chunk[fuel_header].astype(str).str.lower() == 'geothermal') & 
                        (chunk[country_header].astype(str).str.lower() == 'philippines')
                    ]
                    if not filtered_chunk.empty:
                        chunk_list.append(filtered_chunk)

            if not chunk_list:
                print("[WARNING] Slicing yielded zero rows for Philippines Geothermal. Broadening search...")
                for chunk in pd.read_csv(self.input_path, chunksize=5000, low_memory=False):
                    if fuel_header in chunk.columns:
                        filtered_chunk = chunk[chunk[fuel_header].astype(str).str.lower() == 'geothermal']
                        if not filtered_chunk.empty:
                            chunk_list.append(filtered_chunk)

            if not chunk_list:
                print("[ERROR] No matching geothermal dataset records located.")
                sys.exit(1)

            df = pd.concat(chunk_list, ignore_index=True)
            print(f"[SUCCESS] Loaded and isolated unique database slice with {df.shape[0]} rows.")
            
            self.raw_data = df
            return self.raw_data

        except Exception as e:
            print(f"[CRITICAL ERROR] Failed during chunk ingestion: {e}")
            sys.exit(1)

    def map_columns_dynamically(self):
        """Helper method to scan the dataset and lock onto valid engineering columns regardless of text casing."""
        if self.raw_data is None or self.raw_data.empty:
            return
            
        cols = self.raw_data.columns
        cols_lower = [c.lower() for c in cols]
        
        if 'capacity_mw' in cols_lower:
            self.target_col = cols[cols_lower.index('capacity_mw')]
        else:
            numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
            self.target_col = numeric_cols[0] if len(numeric_cols) > 0 else cols[0]
            
        if 'name' in cols_lower:
            self.label_col = cols[cols_lower.index('name')]
        else:
            self.label_col = cols[0]
            
        if 'owner' in cols_lower:
            self.group_col = cols[cols_lower.index('owner')]
        elif 'source' in cols_lower:
            self.group_col = cols[cols_lower.index('source')]
        else:
            self.group_col = cols[-1]

        print(f"[METADATA] Auto-mapped targets -> Metric: '{self.target_col}', Label: '{self.label_col}', Grouping: '{self.group_col}'")

    def clean_data(self) -> pd.DataFrame:
        """Step 2: Automated Data Pipeline & Cleaning."""
        print("\n[INFO] Starting Automated Cleaning Pipeline...")
        if self.raw_data is None or self.raw_data.empty:
            print("[ERROR] No data available to clean.")
            return None

        try:
            df = self.raw_data.copy()
            self.map_columns_dynamically()

            if self.target_col in df.columns:
                median_val = df[self.target_col].median()
                df[self.target_col] = df[self.target_col].fillna(median_val)

            df = df.drop_duplicates()
            self.cleaned_data = df
            
            clean_output_path = os.path.join("data", "dataset_cleaned.csv")
            df.to_csv(clean_output_path, index=False)
            print(f"[SUCCESS] Cleaned unique data slice exported to: {clean_output_path}")
            return self.cleaned_data

        except Exception as e:
            print(f"[ERROR] Exception during cleaning pipeline: {e}")
            return self.raw_data

    def perform_engineering_analytics(self):
        """Step 3: Engineering Data Analytics utilizing strict NumPy computational engines."""
        print("\n[INFO] Executing Engineering Analytics Engine...")
        if self.cleaned_data is None or self.cleaned_data.empty:
            print("[ERROR] Cleaned dataset unavailable.")
            return

        try:
            df = self.cleaned_data
            data_vector = pd.to_numeric(df[self.target_col], errors='coerce').dropna().to_numpy(dtype=np.float64)
            
            if len(data_vector) == 0:
                print("[ERROR] Target column contains no computational numeric records.")
                return

            self.metrics['mean'] = np.mean(data_vector)
            self.metrics['median'] = np.median(data_vector)
            self.metrics['std_dev'] = np.std(data_vector)
            self.metrics['variance'] = np.var(data_vector)

            deviation = data_vector - self.metrics['mean']
            if self.metrics['std_dev'] > 0:
                self.metrics['skewness'] = np.mean(deviation ** 3) / (self.metrics['std_dev'] ** 3)
            else:
                self.metrics['skewness'] = 0.0

            print(f"--- ENGINEERING STATISTICAL MATRIX FOR: {self.target_col} ---")
            print(f"Computed Mean         : {self.metrics['mean']:.4f} MW")
            print(f"Computed Median       : {self.metrics['median']:.4f} MW")
            print(f"Standard Deviation    : {self.metrics['std_dev']:.4f}")
            print(f"Variance Matrix       : {self.metrics['variance']:.4f}")
            print(f"Calculated Skewness   : {self.metrics['skewness']:.4f}")
            
            if self.group_col in df.columns:
                print(f"\n[COMPARATIVE ANALYSIS] Segmented via: {self.group_col}")
                unique_groups = df[self.group_col].dropna().unique()
                for grp in unique_groups[:3]:
                    sub_df = df[df[self.group_col] == grp]
                    sub_vector = pd.to_numeric(sub_df[self.target_col], errors='coerce').dropna().to_numpy(dtype=np.float64)
                    if len(sub_vector) > 0:
                        print(f" -> Group Subset [{grp}] Mean Capacity: {np.mean(sub_vector):.2f} MW")

        except Exception as e:
            print(f"[ERROR] Analytics computation failure: {e}")

    def generate_static_plots(self):
        """Step 4: Generation of 3 Mandatory Static Visualizations."""
        print("\n[INFO] Rendering 3 Static Engineering Visualizations...")
        if self.cleaned_data is None or self.cleaned_data.empty:
            return

        try:
            df = self.cleaned_data
            sns.set_theme(style="darkgrid")

            # Chart 1: Distribution Profile (Histogram)
            plt.figure(figsize=(7, 4.5))
            sns.histplot(pd.to_numeric(df[self.target_col], errors='coerce').dropna(), kde=True, color="darkgreen")
            plt.title(f"Statistical Density Profile: {self.target_col}")
            plt.xlabel("Capacity (MW)")
            plt.savefig(os.path.join(self.output_dir, "static_distribution_histogram.png"), dpi=300)
            plt.close()

            # Chart 2: Statistical Boundary Limits (Boxplot)
            plt.figure(figsize=(6, 4.5))
            sns.boxplot(x=pd.to_numeric(df[self.target_col], errors='coerce'), color="cyan")
            plt.title(f"Operational Boundary Analysis: {self.target_col}")
            plt.xlabel("Capacity (MW)")
            plt.savefig(os.path.join(self.output_dir, "static_variability_boxplot.png"), dpi=300)
            plt.close()

            # Chart 3: System Scaling (Scatter Plot)
            plt.figure(figsize=(8, 5))
            chart_df = df.copy()
            chart_df[self.target_col] = pd.to_numeric(chart_df[self.target_col], errors='coerce')
            sns.scatterplot(data=chart_df, x=self.label_col, y=self.target_col, s=120, color="purple")
            plt.title("Operational Capacity Profile Mapping")
            plt.xticks(rotation=35, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "static_correlation_scatter.png"), dpi=300)
            plt.close()

            print(f"[SUCCESS] 3 static figures written into '{self.output_dir}'")
        except Exception as e:
            print(f"[ERROR] Failed to compile static figures: {e}")

    def generate_animated_plots(self):
        """Step 5: Generation of 2 Mandatory Animated Graphs."""
        print("\n[INFO] Initializing Animation Generation Engine...")
        if self.cleaned_data is None or self.cleaned_data.empty:
            return

        try:
            df = self.cleaned_data.copy().reset_index(drop=True)
            df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
            df = df.dropna(subset=[self.target_col])
            num_records = len(df)
            
            # --- ANIMATION 1: Trend Sweep Profile ---
            print("[INFO] Rendering Animation 1/2: System Trend Sweep...")
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            x_vals1, y_vals1 = [], []
            ln1, = ax1.plot([], [], 'g-o', lw=2, markersize=8, color="firebrick")
            
            ax1.set_xlim(-0.5, num_records - 0.5)
            ax1.set_ylim(0, float(df[self.target_col].max() * 1.2))
            ax1.set_title("Real-Time Telemetry Trend Sweep")
            ax1.set_xlabel("Plant Record Index")
            ax1.set_ylabel("Capacity (MW)")

            def init1():
                ln1.set_data([], [])
                return ln1,

            def update1(frame):
                x_vals1.append(frame)
                y_vals1.append(float(df[self.target_col].iloc[frame]))
                ln1.set_data(x_vals1, y_vals1)
                return ln1,

            ani1 = animation.FuncAnimation(fig1, update1, frames=num_records, init_func=init1, blit=True, repeat=False)
            anim_path1 = os.path.join(self.output_dir, "animated_system_trend.gif")
            ani1.save(anim_path1, writer='pillow', fps=2)
            plt.close(fig1)
            print(f"[SUCCESS] Animation 1 compiled at '{anim_path1}'")

            # --- ANIMATION 2: Cumulative Growth Profile ---
            print("[INFO] Rendering Animation 2/2: Cumulative Grid Expansion...")
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            
            # Use numpy engine to compute cumulative step array
            cumulative_data = np.cumsum(df[self.target_col].to_numpy())
            
            x_vals2, y_vals2 = [], []
            ln2, = ax2.plot([], [], 'b-s', lw=2, markersize=8, color="darkblue")
            
            ax2.set_xlim(-0.5, num_records - 0.5)
            ax2.set_ylim(0, float(cumulative_data[-1] * 1.1))
            ax2.set_title("Aggregate Geothermal Grid Capacity Growth Profile")
            ax2.set_xlabel("Cumulative Plant Additions")
            ax2.set_ylabel("Total Grid Energy Contribution (MW)")

            def init2():
                ln2.set_data([], [])
                return ln2,

            def update2(frame):
                x_vals2.append(frame)
                y_vals2.append(float(cumulative_data[frame]))
                ln2.set_data(x_vals2, y_vals2)
                return ln2,

            ani2 = animation.FuncAnimation(fig2, update2, frames=num_records, init_func=init2, blit=True, repeat=False)
            anim_path2 = os.path.join(self.output_dir, "animated_cumulative_growth.gif")
            ani2.save(anim_path2, writer='pillow', fps=2)
            plt.close(fig2)
            print(f"[SUCCESS] Animation 2 compiled at '{anim_path2}'")
            
        except Exception as e:
            print(f"[ERROR] Animation rendering failure: {e}")


if __name__ == "__main__":
    INPUT_PATH = "data/dataset_original.csv"
    OUTPUT_PATH = "outputs/"

    pipeline = GeothermalPipeline(input_path=INPUT_PATH, output_dir=OUTPUT_PATH)
    
    print("=================================================================")
    print("         LAUNCHING GEOTHERMAL PIPELINE ANALYTICS SUITE          ")
    print("=================================================================")
    
    pipeline.ingest_data()
    pipeline.clean_data()
    
    if pipeline.cleaned_data is not None and not pipeline.cleaned_data.empty:
        pipeline.perform_engineering_analytics()
        pipeline.generate_static_plots()
        pipeline.generate_animated_plots()
        
        print("\n=================================================================")
        print(" [PIPELINE SUCCESS] Process Complete. Check 'outputs/' directory.")
        print("=================================================================")