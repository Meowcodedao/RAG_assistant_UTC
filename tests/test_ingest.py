from app.services.corpus import slugify, unique_id

def test_slugify_strips_vietnamese_diacritics():
    assert slugify("Bài 5.pdf")=="bai_5"
    assert slugify("Đồ thị.pdf")=="do_thi"
    assert slugify("Bai8_Recurrent neural network.pdf")=="bai8_recurrent_neural_network"

def test_unique_id_avoids_collisions():
    used=set()
    assert unique_id("bai_5",used)=="bai_5"
    assert unique_id("bai_5",used)=="bai_5_2"
