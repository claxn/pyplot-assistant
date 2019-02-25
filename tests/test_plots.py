
def test_hist():
    import numpy as np
    from pyplot_assistant.pyplot_assistant import PyPlotHist
    plt_hist = PyPlotHist(normed=True, xlabel="Height")
    plt_hist.plot([np.random.rand(1000).tolist()])
    plt_hist.show()


def test_line():
    import numpy as np
    from pyplot_assistant.pyplot_assistant import PyPlotHist
    plt_hist = PyPlotHist(normed=True, xlabel="Height")
    plt_hist.plot([np.random.rand(1000).tolist()])
    plt_hist.show()

test_line()