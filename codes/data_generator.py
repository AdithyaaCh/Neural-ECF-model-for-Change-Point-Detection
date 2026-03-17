import torch
import numpy as np
from student_t import StudentTConfig, sample_student_t_series
from sub_gaussian import SubGaussianConfig, sample_subgaussian_series

class TripletDataGenerator:
    def __init__(self, p=50, L=200):
        """
        p: Number of dimensions (e.g., 50)
        L: Window length
        """
        self.p = p
        self.L = L

    def _get_random_rhos(self, min_shift=0.3):
        """
        Generates omnidirectional rho shifts. 
        Guarantees that rho_pre and rho_post are different by at least 'min_shift'.
        This teaches the model about subtle (0.6 -> 0.1) and violent (0.8 -> -0.8) shifts.
        """
        rho_pre = np.random.uniform(-0.85, 0.85)
        rho_post = np.random.uniform(-0.85, 0.85)
        
        # Keep rolling until the shift is mathematically significant (this is assuming that the teaching the model about highly subtle changes will only hallucinate it)
        while abs(rho_pre - rho_post) < min_shift:
            rho_post = np.random.uniform(-0.85, 0.85)
        return rho_pre, rho_post

    def _generate_single_series(self):
        """Randomly generates a full series with dynamic distributional parameters."""
        
        dist_type = np.random.choice(['student_t', 'subgaussian'])
        rho_pre, rho_post = self._get_random_rhos(min_shift=0.3)
        
        if dist_type == 'student_t':
            # Jitter the degrees of freedom (nu) between 2.5 and 10.0
            # This teaches the model to recognize changes in tail-heaviness
            nu_pre = np.random.uniform(2.5, 10.0)
            nu_post = np.random.uniform(2.5, 10.0)
            
            config = StudentTConfig(
                n_samples=2000, n_star=1000, p=self.p,
                rho_pre=rho_pre, rho_post=rho_post,
                nu_pre=nu_pre, nu_post=nu_post
            )
            X, n_star = sample_student_t_series(config)
            
        else:
            # Jitter the stability parameter (alpha) between 1.5 and 1.98
            # This handles the fractional moment changes tested in the benchmark
            alpha_pre = np.random.uniform(1.5, 1.98)
            alpha_post = np.random.uniform(1.5, 1.98)
            
            config = SubGaussianConfig(
                n_samples=2000, n_star=1000, p=self.p,
                rho_pre=rho_pre, rho_post=rho_post,
                alpha_pre=alpha_pre, alpha_post=alpha_post
            )
            X, n_star = sample_subgaussian_series(config)
            
        return X, n_star

    def generate_triplet_batch(self, batch_size=32):
        """
        Generates (Anchor, Positive, Hard-Negative) triplets.
        """
        anchors, positives, negatives = [], [], []

        for _ in range(batch_size):
            X, n_star = self._generate_single_series()
            
            # 1. Anchor: Randomly sample anywhere in Regime 1
            idx_a = np.random.randint(0, n_star - self.L - 20)
            anchors.append(X[idx_a : idx_a + self.L])
            
            # 2. Positive: Overlapping window to teach the model that temporal 
            # shifting within the same regime should output the exact same ECF
            idx_p = idx_a + np.random.randint(5, 20)
            positives.append(X[idx_p : idx_p + self.L])
            
            # 3. Hard & Soft Negatives (50/50 split)
            # This teaches the model the difference between a sharp boundary and a distant regime
            if np.random.rand() > 0.5:
                # Hard Negative: Right on the other side of the change point
                idx_n = n_star + np.random.randint(0, 10)
            else:
                # Soft Negative: Deep into Regime 2
                idx_n = np.random.randint(n_star + 50, len(X) - self.L)
                
            negatives.append(X[idx_n : idx_n + self.L])

        return (torch.tensor(np.array(anchors), dtype=torch.float32),
                torch.tensor(np.array(positives), dtype=torch.float32),
                torch.tensor(np.array(negatives), dtype=torch.float32))