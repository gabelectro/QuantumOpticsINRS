from TimeTagger import createTimeTagger, Countrate
tt = createTimeTagger()

for ch in (4, -4):                      # +4 = rising edge, -4 = falling edge
    for level in [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2,
                   0.2,  0.4,  0.6,  0.8,  1.0,  1.2, 1.6, 2.0]:
        tt.setTriggerLevel(4, level)    # always the physical channel number
        cr = Countrate(tt, [ch]); cr.startFor(int(0.2e12)); cr.waitUntilFinished()
        print(f"ch {ch:+d}  {level:+.1f} V  ->  {cr.getData()[0]:>14,.0f} /s")