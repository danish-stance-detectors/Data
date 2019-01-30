Contains all Reddit submissions in JSON format in the following format:

{
    Submission-info,
    { Author-info },
    { Subreddit-info },
    Submission-comments: [
        {
            comment1,
            { user1 },
            comment2,
            { user 2},
            ...,
            commentN,
            { userN }
        }
    ]
}

Note that a comment may be deleted. This is identified by the "is_deleted" property.
Such comments have the body text "[deleted]". These comments are kept, 
as nested replies may provide relevant information.