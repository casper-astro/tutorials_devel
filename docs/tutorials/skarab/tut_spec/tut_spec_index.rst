Tutorial 4: Wideband Spectrometer
========================================================

The SKARAB wideband spectrometer tutorials make use of a SKARAB ADC Board. There are two different bandwidth modes available for a SKARAB ADC Board depending on which ADC module is on the board. These are detailed in the table below:

+----------------+---------------------+--------------------+---------------------------------------+
| Bandwidth Mode | Sampling Frequency  | Decimation Factor  | SKARAB ADC Support                    |
+================+=====================+====================+=======================================+
|DDC Mode        | 3 GSPS              | 4                  | All SKARAB ADC Board Types            |
+----------------+---------------------+--------------------+---------------------------------------+
|Bypass Mode     | 2.8 GSPS            | None (bypassed)    | Only SKARAB ADC Boards with ADC32RF45 |
+----------------+---------------------+--------------------+---------------------------------------+

As such, there are two versions of wideband spectrometer tutorial, one for each mode. Links to these are given below:

1. :doc:`Wideband Spectrometer - DDC Mode <tut_spec>`
2. :doc:`Wideband Spectrometer - Bypass Mode <tut_spec_byp>`

For more information on working with the SKARAB ADC Board please see `Tutorial 5: SKARAB ADC Synchronous Data Acquisition <https://casper-toolflow.readthedocs.io/projects/tutorials/en/latest/tutorials/skarab/tut_adc/tut_adc_index.html>`__

.. toctree::
    :maxdepth: 2
    :hidden:

    tut_spec
    tut_spec_byp
