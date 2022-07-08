"""Test for reference worker."""

from dipcoatimage.finitedepth import SubstrateReference
from dipcoatimage.finitedepth_gui.inventory import StructuredReferenceArgs
from dipcoatimage.finitedepth_gui.workers import ReferenceWorker


def test_ReferenceWorker_setStructuredReferenceArgs(qtbot):
    worker = ReferenceWorker()
    assert worker.referenceType() is None
    assert worker.image().size == 0
    assert worker.parameters() is None
    assert worker.drawOptions() is None

    valid_data1 = StructuredReferenceArgs(
        SubstrateReference, (0, 0, None, None), (0, 0, None, None), None, None
    )
    worker.setStructuredReferenceArgs(valid_data1)
    assert worker.referenceType() == valid_data1.type
    assert worker.parameters() == worker.referenceType().Parameters()
    assert worker.drawOptions() == worker.referenceType().DrawOptions()

    valid_data2 = StructuredReferenceArgs(
        SubstrateReference,
        (0, 0, None, None),
        (0, 0, None, None),
        SubstrateReference.Parameters(),
        SubstrateReference.DrawOptions(),
    )
    worker.setStructuredReferenceArgs(valid_data2)
    assert worker.referenceType() == valid_data2.type
    assert worker.parameters() == valid_data2.parameters
    assert worker.drawOptions() == valid_data2.draw_options

    type_invalid_data = StructuredReferenceArgs(
        type,
        (0, 0, None, None),
        (0, 0, None, None),
        SubstrateReference.Parameters(),
        SubstrateReference.DrawOptions(),
    )
    worker.setStructuredReferenceArgs(type_invalid_data)
    assert worker.referenceType() is None
    assert worker.parameters() is None
    assert worker.drawOptions() is None
