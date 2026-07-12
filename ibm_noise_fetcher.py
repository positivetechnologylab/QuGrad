#!/usr/bin/env python3
"""
IBM Noise Model Fetcher

Collects and caches IBM quantum backend noise models for use in noisy training experiments.
Note: the IBM Runtime API only exposes the *current* calibration. To build a
multi-day cache you must run this collector each day (ideally shortly after the
daily 12am ET calibration) so that the requested dates are populated over time.
"""

import os
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_aer.noise import NoiseModel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    NoiseModel = None  # Define NoiseModel as None when import fails
    print("Warning: qiskit_ibm_runtime not available. Install with: pip install qiskit-ibm-runtime")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBMNoiseModelFetcher:
    """
    Fetches and caches IBM quantum backend noise models.

    Noise models are cached locally to avoid repeated API calls and ensure
    reproducibility of experiments across different runs.
    """

    # Mapping for decommissioned backends to their replacements
    DECOMMISSIONED_BACKENDS = {
        'ibm_brisbane': 'ibm_marrakesh',  # Brisbane decommissioned, use Marrakesh (both 156 qubits)
    }

    def __init__(self, cache_dir: str = "noise_models"):
        """
        Initialize the noise model fetcher.

        Args:
            cache_dir: Directory to store cached noise models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if not QISKIT_AVAILABLE:
            raise ImportError("qiskit_ibm_runtime required. Install: pip install qiskit-ibm-runtime")

        # Initialize IBM Quantum service (optional - only needed for fetching new models)
        # If loading cached models only, service is not required
        self.service = None
        try:
            # Check for direct token via env var (bypass ~/.qiskit config if needed)
            token = os.environ.get('QISKIT_IBM_TOKEN')
            if token:
                logger.info("Using token from QISKIT_IBM_TOKEN environment variable")
                # Try platform first, then quantum
                try:
                    self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
                except:
                    try:
                        self.service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                    except:
                        # Maybe cloud?
                        self.service = QiskitRuntimeService(channel="ibm_cloud", token=token)
            else:
                self.service = QiskitRuntimeService()
                
            logger.info("Successfully initialized QiskitRuntimeService")
        except Exception as e:
            logger.warning(f"QiskitRuntimeService not available: {e}")
            logger.warning("Will use cached noise models only (fetching new models disabled)")
    
    def get_available_backends(self) -> List[str]:
        """Get list of available IBM quantum backends."""
        try:
            backends = self.service.backends()
            backend_names = [backend.name for backend in backends if backend.operational]
            logger.info(f"Found {len(backend_names)} operational backends")
            return backend_names
        except Exception as e:
            logger.error(f"Failed to get available backends: {e}")
            return []
    
    def collect_noise_models(self, backends: List[str], days: int = 5) -> Dict[str, List[str]]:
        """
        Collect noise models from specified backends over multiple days.
        
        Args:
            backends: List of backend names to collect from
            days: Number of days to collect (starting from today going backwards)
            
        Returns:
            Dictionary mapping backend names to list of collected dates
        """
        collected = {}
        missing = {}
        
        # Generate date list (going backwards from today)
        dates = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Collecting noise models for {len(backends)} backends over {days} days")
        logger.info(f"Backends: {backends}")
        logger.info(f"Dates: {dates}")
        
        for backend_name in backends:
            collected[backend_name] = []
            missing[backend_name] = []
            backend_dir = self.cache_dir / backend_name
            backend_dir.mkdir(exist_ok=True)
            
            for date_str in dates:
                try:
                    # Check if already cached
                    model_file = backend_dir / f"{date_str}_model.pkl"
                    metadata_file = backend_dir / f"{date_str}.json"
                    
                    if model_file.exists() and metadata_file.exists():
                        logger.info(f"✓ {backend_name} {date_str} already cached")
                        collected[backend_name].append(date_str)
                        continue
                    
                    # IBM APIs do not expose historical calibrations.  Only the
                    # current day's calibration can be fetched.  If a requested
                    # date is not cached yet and it is not today, warn the user
                    # so they can re-run the collector on that day.
                    if date_str != today_str:
                        logger.warning(
                            "Historical noise models are not available via Qiskit. "
                            f"Skipping {backend_name} {date_str}. Re-run the collector on "
                            "the desired day to capture that calibration."
                        )
                        missing[backend_name].append(date_str)
                        continue
                    
                    # Fetch backend and create noise model
                    logger.info(f"Fetching {backend_name} noise model for {date_str}...")
                    backend = self.service.backend(backend_name)
                    
                    # Create noise model from backend
                    noise_model = NoiseModel.from_backend(backend)
                    
                    # Save noise model
                    with open(model_file, 'wb') as f:
                        pickle.dump(noise_model, f)
                    
                    # Save metadata
                    try:
                        config = backend.configuration()
                        coupling_map = getattr(config, 'coupling_map', None)
                        num_qubits = getattr(config, 'num_qubits', getattr(backend, 'num_qubits', None))
                    except Exception:
                        coupling_map = None
                        num_qubits = getattr(backend, 'num_qubits', None)

                    metadata = {
                        'backend_name': backend_name,
                        'date': date_str,
                        'collected_at': datetime.now().isoformat(),
                        'num_qubits': num_qubits,
                        'coupling_map': coupling_map,
                        'basis_gates': list(noise_model.basis_gates),
                        'noise_instructions': len(noise_model._local_quantum_errors) + len(noise_model._default_quantum_errors)
                    }
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    collected[backend_name].append(date_str)
                    logger.info(f"✓ Successfully collected {backend_name} {date_str}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed to collect {backend_name} {date_str}: {e}")
                    missing[backend_name].append(date_str)
                    continue
        
        # Save collection log
        log_file = self.cache_dir / "collection_log.json"
        collection_log = {
            'collected_at': datetime.now().isoformat(),
            'backends_requested': backends,
            'days_requested': days,
            'dates_attempted': dates,
            'results': collected,
            'missing': missing,
            'total_models': sum(len(dates) for dates in collected.values())
        }
        
        with open(log_file, 'w') as f:
            json.dump(collection_log, f, indent=2)
        
        logger.info(f"Collection complete. Total models: {collection_log['total_models']}")
        return collected
    
    def get_noise_model(self, backend_name: str, date: str) -> Optional[NoiseModel]:
        """
        Load a cached noise model for specific backend and date.
        
        Args:
            backend_name: Name of the IBM backend
            date: Date string in YYYY-MM-DD format
            
        Returns:
            NoiseModel if found, None otherwise
        """
        model_file = self.cache_dir / backend_name / f"{date}_model.pkl"
        
        if not model_file.exists():
            logger.error(f"Noise model not found: {model_file}")
            return None
        
        try:
            with open(model_file, 'rb') as f:
                noise_model = pickle.load(f)
            logger.info(f"Loaded noise model: {backend_name} {date}")
            return noise_model
        except Exception as e:
            logger.error(f"Failed to load noise model {model_file}: {e}")
            return None
    
    def list_cached_models(self) -> Dict[str, List[str]]:
        """
        List all cached noise models.
        
        Returns:
            Dictionary mapping backend names to list of available dates
        """
        cached = {}
        
        if not self.cache_dir.exists():
            return cached
        
        for backend_dir in self.cache_dir.iterdir():
            if backend_dir.is_dir():
                backend_name = backend_dir.name
                dates = []
                
                for model_file in backend_dir.glob("*_model.pkl"):
                    date_str = model_file.stem.replace("_model", "")
                    dates.append(date_str)
                
                if dates:
                    cached[backend_name] = sorted(dates)
        
        return cached
    
    def validate_cached_models(self) -> Dict[str, Dict[str, bool]]:
        """
        Validate all cached noise models can be loaded.
        
        Returns:
            Dictionary mapping backend -> date -> validation_success
        """
        cached_models = self.list_cached_models()
        validation_results = {}
        
        for backend_name, dates in cached_models.items():
            validation_results[backend_name] = {}
            
            for date in dates:
                try:
                    noise_model = self.get_noise_model(backend_name, date)
                    validation_results[backend_name][date] = noise_model is not None
                except Exception as e:
                    logger.error(f"Validation failed for {backend_name} {date}: {e}")
                    validation_results[backend_name][date] = False
        
        return validation_results


def main():
    """Command line interface for noise model collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect IBM quantum backend noise models")
    parser.add_argument('--backends', type=str, 
                       default='ibm_brisbane,ibm_torino,ibm_fez,ibm_marrakesh,ibm_kyoto',
                       help='Comma-separated list of backend names')
    parser.add_argument('--days', type=int, default=5,
                       help='Number of days to collect (default: 5). Run this '
                            'script daily to populate historical dates.')
    parser.add_argument('--list', action='store_true',
                       help='List cached models')
    parser.add_argument('--validate', action='store_true',
                       help='Validate cached models')
    
    args = parser.parse_args()
    
    fetcher = IBMNoiseModelFetcher()
    
    if args.list:
        cached = fetcher.list_cached_models()
        print("\nCached Noise Models:")
        print("=" * 50)
        for backend, dates in cached.items():
            print(f"{backend}: {len(dates)} models ({', '.join(dates)})")
        return
    
    if args.validate:
        results = fetcher.validate_cached_models()
        print("\nValidation Results:")
        print("=" * 50)
        for backend, date_results in results.items():
            valid_count = sum(date_results.values())
            total_count = len(date_results)
            print(f"{backend}: {valid_count}/{total_count} valid")
        return
    
    # Collect noise models
    backends = [b.strip() for b in args.backends.split(',')]
    collected = fetcher.collect_noise_models(backends, args.days)
    
    print("\nCollection Summary:")
    print("=" * 50)
    for backend, dates in collected.items():
        print(f"{backend}: {len(dates)} models collected")


if __name__ == "__main__":
    main()
