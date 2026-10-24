

class TestLatentImplicationTesting:
    def test_skips_unobserved_latent_implications(self):
        rng = np.random.default_rng(3)
        n = 200
        a = rng.normal(size=n)
        w = rng.normal(size=n)
        zz = 0.7 * w + rng.normal(scale=0.3, size=n)
        x = 0.4 * a + 0.9 * zz + rng.normal(scale=0.3, size=n)
        y = 0.6 * x + 1.5 * a + rng.normal(scale=0.3, size=n)
        data = pd.DataFrame({"A": a, "W": w, "X": x, "Y": y})

        model = pathmc.model("ZZ ~ W\nX ~ A + ZZ\nY ~ X", data=data, latent=["ZZ"])

        result = model.test_implications().to_dataframe()
        assert not ((result["x"] == "A") & (result["y"] == "ZZ")).any()
        assert ((result["x"] == "A") & (result["y"] == "Y")).any()

