$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview (Handles both images and PDFs)
    function readFile(input) {
        if (input.files && input.files[0]) {
            var file = input.files[0];
            var reader = new FileReader();
            
            if (file.type.includes("image")) {
                reader.onload = function (e) {
                    $('#uploadedImage').attr('src', e.target.result).show(); // Show image preview
                    $('#pdfPreview').hide(); // Hide PDF preview
                    $('.image-section').show();
                };
                reader.readAsDataURL(file);
            } else if (file.type === "application/pdf") {
                reader.onload = function () {
                    var url = URL.createObjectURL(file); // Create object URL for PDF
                    $('#pdfPreview').attr('src', url).show(); // Show PDF preview
                    $('#uploadedImage').hide(); // Hide image preview
                    $('.image-section').show();
                };
                reader.readAsArrayBuffer(file);
            }
        }
    }

    
    // Trigger file input when custom button is clicked
    $("#customUploadButton").click(function () {
        $("#imageUpload").click();
    });

    // Handle file selection
    $("#imageUpload").change(function () {
        $('#btn-predict').show();
        $('#result').text('').hide();
        readFile(this);

        // Show file name inside button
        if (this.files.length > 0) {
            $('#customUploadButton').text(this.files[0].name);
        }
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling API /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Hide loader and show result
                $('.loader').hide();
                $('#result').fadeIn(600).html('<span style="color:white;">' + data + '</span>');
                console.log('Success!');
            },
            error: function () {
                $('.loader').hide();
                $('#result').fadeIn(600).html('<span style="color:red;">Error processing the file</span>');
            }
        });
    });
});
